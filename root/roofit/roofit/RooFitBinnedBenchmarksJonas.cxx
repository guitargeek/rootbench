#include "RooWorkspace.h"
#include "RooAddPdf.h"
#include "RooRealVar.h"
#include "RooMinimizer.h"
#include "TFile.h"
#include "TH1.h"
#include "TRandom.h"
#include "TError.h"
#include "RooStats/HistFactory/Measurement.h"
#include "RooStats/HistFactory/HistoToWorkspaceFactoryFast.h"
#include "RooStats/ModelConfig.h"
#include "RooRealSumPdf.h"
#include "RooRandom.h"

#include "benchmark/benchmark.h"

using namespace RooFit;
using namespace RooStats;
using namespace HistFactory;

namespace {
  constexpr bool verbose = false;

  // test matrix configuration
  const std::vector<int> nChannelsVector = {1, 2, 3};
  const std::vector<int> nBinsVector {5, 10, 15};
  const int nBinsForChannelScan = 10;
  const int nChannelsForBinScan = 1;
  const std::vector<int> nCPUVector {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

  constexpr std::size_t nParamSets = 3;

  auto const timeUnit = benchmark::kMillisecond;

  void setupRooMsgService() {
     RooMsgService::instance().setGlobalKillBelow(RooFit::FATAL);
     RooMsgService::instance().getStream(1).removeTopic(RooFit::Minimization);
     RooMsgService::instance().getStream(1).removeTopic(RooFit::NumIntegration);
     RooMsgService::instance().getStream(1).removeTopic(RooFit::Eval);
  }

} // namespace

Sample addVariations(Sample asample, int nnps, bool channel_crosstalk, int channel)
{
   for (int nuis = 0; nuis < nnps; ++nuis) {
      TRandom *R = new TRandom(channel * nuis / nnps);
      Double_t random = R->Rndm();
      double uncertainty_up = (1 + random) / sqrt(100);
      double uncertainty_down = (1 - random) / sqrt(100);
      if(verbose) {
        std::cout << "in channel " << channel << "nuisance +/- [" << uncertainty_up << "," << uncertainty_down << "]"
                  << std::endl;
      }
      std::string nuis_name = "norm_uncertainty_" + std::to_string(nuis);
      if (!channel_crosstalk) {
         nuis_name = nuis_name + "_channel_" + std::to_string(channel);
      }
      asample.AddOverallSys(nuis_name, uncertainty_up, uncertainty_down);
   }
   return asample;
}

Channel makeChannel(int channel, int nbins, int nnps)
{
   std::string channel_name = "Region" + std::to_string(channel);
   Channel chan(channel_name);
   gDirectory = nullptr;
   auto Signal_Hist = new TH1F("Signal", "Signal", nbins, 0, nbins);
   auto Background_Hist = new TH1F("Background", "Background", nbins, 0, nbins);
   auto Data_Hist = new TH1F("Data", "Data", nbins, 0, nbins);
   for (Int_t bin = 1; bin <= nbins; ++bin) {
      for (Int_t i = 0; i <= bin; ++i) {
         Signal_Hist->Fill(bin + 0.5);
         Data_Hist->Fill(bin + 0.5);
      }
      for (Int_t i = 0; i <= nbins; ++i) {
         Background_Hist->Fill(bin + 0.5);
         Data_Hist->Fill(bin + 0.5);
      }
   }
   chan.SetData(Data_Hist);
   Sample background("background");
   background.SetNormalizeByTheory(false);
   background.SetHisto(Background_Hist);
   background.ActivateStatError();
   Sample signal("signal");
   signal.SetNormalizeByTheory(false);
   signal.SetHisto(Signal_Hist);
   signal.ActivateStatError();
   signal.AddNormFactor("SignalStrength", 1, 0, 3);

   if (nnps > 0) {
      signal = addVariations(signal, nnps, true, channel);
      background = addVariations(background, nnps, false, channel);
   }
   chan.AddSample(background);
   chan.AddSample(signal);
   return chan;
}

void buildBinnedTest(int n_channels = 1, int nbins = 10, int nnps = 1, const char *name_rootfile = "")
{
   if(verbose) {
     std::cout << "in build binned test with output" << name_rootfile << std::endl;
   }
   Measurement meas("meas", "meas");
   meas.SetPOI("SignalStrength");
   meas.SetLumi(1.0);
   meas.SetLumiRelErr(0.10);
   meas.AddConstantParam("Lumi");
   Channel chan;
   for (int channel = 0; channel < n_channels; ++channel) {
      chan = makeChannel(channel, nbins, nnps);
      meas.AddChannel(chan);
   }
   HistoToWorkspaceFactoryFast hist2workspace(meas);
   RooWorkspace *ws;
   if (n_channels < 2) {
      ws = hist2workspace.MakeSingleChannelModel(meas, chan);
   } else {
      ws = hist2workspace.MakeCombinedModel(meas);
   }
   RooFIter iter = ws->components().fwdIterator();
   RooAbsArg *arg;
   while ((arg = iter.next())) {
      if (arg->IsA() == RooRealSumPdf::Class()) {
         arg->setAttribute("BinnedLikelihood");
         if(verbose) std::cout << "component " << arg->GetName() << " is a binned likelihood" << std::endl;
      }
   }
   ws->SetName("BinnedWorkspace");
   ws->writeToFile(name_rootfile);
}


// right now, this is duplicated from vectorisedPDFs/benchAddPdf.cxx
void randomiseParameters(const RooArgSet& parameters, ULong_t seed=0) {
  auto random = RooRandom::randomGenerator();
  if (seed != 0)
    random->SetSeed(seed);

  for (auto param : parameters) {
    auto par = static_cast<RooAbsRealLValue*>(param);
    const double uni = random->Uniform();
    const double min = par->getMin();
    const double max = par->getMax();
    par->setVal(min + uni*(max-min));
  }
}

//############## End of Base Algorithms ##############################
//####################################################################
//############## Start Of # Tests #############################

static void BM_RooFit_BinnedLikelihood(benchmark::State &state)
{
   gErrorIgnoreLevel = kInfo;
   setupRooMsgService();
   int chan = state.range(0);
   int nbins = state.range(1);
   int cpu = state.range(2);
   bool batchMode = state.range(3);
   auto infile = std::make_unique<TFile>("workspace.root", "RECREATE");
   //   if (infile->IsZombie()) {
   buildBinnedTest(chan, nbins, 2, "workspace.root");
   if(verbose) std::cout << "Workspace for tests was created!" << std::endl;
   //}
   infile.reset(TFile::Open("workspace.root"));
   RooWorkspace *w = static_cast<RooWorkspace *>(infile->Get("BinnedWorkspace"));
   std::unique_ptr<RooAbsData> data{w->data("obsData")};
   std::unique_ptr<ModelConfig> mc{static_cast<ModelConfig *>(w->genobj("ModelConfig"))};
   std::unique_ptr<RooAbsPdf> pdf{w->pdf(mc->GetPdf()->GetName())};
   std::unique_ptr<RooAbsReal> nll{pdf->createNLL(*data, NumCPU(cpu, 0), BatchMode(batchMode))};

   RooArgSet parameters;
   pdf->getParameters(data->get(), parameters);

   std::array<RooArgSet, nParamSets> paramSets;
   unsigned int seed = 1337;
   for (auto& paramSet : paramSets) {
     randomiseParameters(parameters, seed++);
     parameters.snapshot(paramSet);
   }

   for (auto _ : state) {
     for (const auto& paramSet : paramSets) {
       parameters = paramSet;
       nll->getVal();
       //for(auto const& arg : parameters) {
         //if(auto param = dynamic_cast<RooRealVar*>(arg)) {
           //double val = param->getVal();
           //param->setVal(val - 1e-4);
           //nll->getVal();
           //param->setVal(val + 1e-4);
           //nll->getVal();
         //}
       //}
     }
   }
}

//############## Run # Tests ###############################

static void ChanArguments(benchmark::internal::Benchmark *b)
{
   for (int nCPU : nCPUVector) {
      //// channel scan
      //for (int nChannels : nChannelsVector) {
         //b->Args({nChannels, nBinsForChannelScan, nCPU});
      //}

      //// bin scan
      //for (int nBins : nBinsVector) {
         //b->Args({nChannelsForBinScan, nBins, nCPU});
      //}
      b->Args({15, 20, nCPU, false});
      b->Args({15, 50, nCPU, true});
   }
}

BENCHMARK(BM_RooFit_BinnedLikelihood)
   ->Apply(ChanArguments)
   ->UseRealTime()
   ->Unit(timeUnit);
   //->Iterations(1);

//############## End Of Tests ########################################
//####################################################################
//############## RUN #################################################

BENCHMARK_MAIN();
