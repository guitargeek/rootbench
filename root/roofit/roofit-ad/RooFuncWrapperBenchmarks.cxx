#include <RooAbsData.h>
#include <RooAbsPdf.h>
#include <RooAddition.h>
#include <RooAddPdf.h>
#include <RooChebychev.h>
#include <RooConstVar.h>
#include <RooDataSet.h>
#include <RooDataHist.h>
#include <RooExponential.h>
#include <RooFuncWrapper.h>
#include <RooGaussian.h>
#include <RooMinimizer.h>
#include <RooProduct.h>
#include <RooRandom.h>
#include <RooRealVar.h>
#include <RooFitResult.h>
#include <RooCategory.h>
#include <RooSimultaneous.h>

#include <RooPlot.h>
#include <TCanvas.h>

#include <TROOT.h>
#include <TSystem.h>
#include <TMath.h>
#include <Math/Factory.h>
#include <Math/Minimizer.h>

#include "BenchmarkUtils.h"

#include "benchmark/benchmark.h"

static int counter = 0;

static void BM_RooFuncWrapper_Minimization(benchmark::State &state)
{

   counter++;

   gInterpreter->ProcessLine("gErrorIgnoreLevel = 2001;");
   auto &msg = RooMsgService::instance();
   msg.setGlobalKillBelow(RooFit::WARNING);

   RooRealVar x("x", "x", 0, -10, 10);

   RooRealVar mu("mu", "mu", 0, -10, 10);
   RooConstVar shift("shift", "shift", 1.0);
   RooAddition muShifted("mu_shifted", "mu_shifted", {mu, shift});

   RooRealVar sigma("sigma", "sigma", 4.0, 0.01, 10);
   RooConstVar scale("scale", "scale", 1.5);
   RooProduct sigmaScaled("sigma_scaled", "sigma_scaled", sigma, scale);

   RooGaussian gauss{"gauss", "gauss", x, muShifted, sigmaScaled};

   RooArgSet normSet{x};

   // Generate the same dataset for all backends.
   RooRandom::randomGenerator()->SetSeed(100);

   std::size_t nEvents = state.range(1);
   std::unique_ptr<RooDataSet> data{gauss.generate(x, nEvents)};

   // Set the values away from the minimum
   mu.setVal(-2.0);
   sigma.setVal(7.0);

   // Save the original parameter values and errors to reset after minimization
   RooArgSet params;
   gauss.getParameters(&normSet, params);
   RooArgSet origParams;
   params.snapshot(origParams);

   std::unique_ptr<RooAbsReal> nllRef{gauss.createNLL(*data, RooFit::BatchMode("off"))};
   std::unique_ptr<RooAbsReal> nllRefBatch{gauss.createNLL(*data, RooFit::BatchMode("cpu"))};
   auto nllRefResolved = static_cast<RooAbsReal *>(nllRefBatch->servers()[0]);

   std::string name = "myNll" + std::to_string(counter);
   RooFuncWrapper nllFunc(name.c_str(), name.c_str(), *nllRefResolved, normSet, data.get());

   std::unique_ptr<RooMinimizer> m = nullptr;
   for (auto _ : state) {
      int code = state.range(0);
      if (code == RooFitADBenchmarksUtils::backend::Reference) {
         m.reset(new RooMinimizer(*nllRef));
      } else if (code == RooFitADBenchmarksUtils::backend::CodeSquashNumDiff) {
         m.reset(new RooMinimizer(nllFunc));
      } else if (code == RooFitADBenchmarksUtils::backend::BatchMode) {
         m.reset(new RooMinimizer(*nllRefBatch));
      } else if (code == RooFitADBenchmarksUtils::backend::CodeSquashAD) {
         RooMinimizer::Config minimizerCfgAd;
         minimizerCfgAd.gradFunc = [&](double *out) { nllFunc.getGradient(out); };
         m.reset(new RooMinimizer(nllFunc, minimizerCfgAd));
      }

      m->setPrintLevel(-1);
      m->setStrategy(0);
      params.assign(origParams);
      m->minimize("Minuit2");
   }
}

std::unique_ptr<RooAbsPdf> createModel(RooRealVar & x, std::string const& channelName)
{
   auto prefix = [&](const char* name) { return name + std::string("_") + channelName; };

   RooRealVar c(prefix("c").c_str(), "c", -0.5, -0.8, 0.2);

   RooExponential expo(prefix("expo").c_str(), "expo", x, c);

   // Create two Gaussian PDFs g1(x,mean1,sigma) anf g2(x,mean2,sigma) and their parameters
   RooRealVar mean1(prefix("mean1").c_str(), "mean of gaussians", 3, 0, 5);
   RooRealVar sigma1(prefix("sigma1").c_str(), "width of gaussians", 2.0, .5, 5.0);
   RooRealVar mean2(prefix("mean2").c_str(), "mean of gaussians", 6, 5, 10);
   RooRealVar sigma2(prefix("sigma2").c_str(), "width of gaussians", 0.5, .1, 2.0);

   RooGaussian sig1(prefix("sig1").c_str(), "Signal component 1", x, mean1, sigma1);
   RooGaussian sig2(prefix("sig2").c_str(), "Signal component 2", x, mean2, sigma2);

   // Build Chebychev polynomial pdf
   RooRealVar a0(prefix("a0").c_str(), "a0", 0.3, 0., 0.5);
   RooRealVar a1(prefix("a1").c_str(), "a1", 0.2, 0., 0.5);
   RooChebychev bkg(prefix("bkg").c_str(), "Background", x, {a0, a1});

   // Sum the composite signal and background
   RooRealVar bkgfrac(prefix("bkgfrac").c_str(), "fraction of background", 0.2, 0.1, 0.25);
   RooRealVar sig1frac(prefix("sig1frac").c_str(), "fraction of signal", 0.2, 0.1, 0.25);
   RooRealVar sig2frac(prefix("sig2frac").c_str(), "fraction of signal", 0.2, 0.1, 0.25);
   RooAddPdf model(prefix("model").c_str(), "g1+g2+a", {bkg, sig1, sig2, expo}, {bkgfrac, sig1frac, sig2frac});

   return std::unique_ptr<RooAbsPdf>{static_cast<RooAbsPdf*>(model.cloneTree())};
}

static void BM_RooFuncWrapper_ManyParams_Minimization(benchmark::State &state)
{
   using namespace RooFit;

   counter++;

   // gInterpreter->ProcessLine("gErrorIgnoreLevel = 2001;");
   // auto &msg = RooMsgService::instance();
   // msg.setGlobalKillBelow(RooFit::WARNING);

   // Generate the same dataset for all backends.
   RooRandom::randomGenerator()->SetSeed(100);

   std::size_t nEvents = state.range(1);

   RooCategory channelCat{"channel_cat", ""};

   std::map<std::string,RooAbsPdf*> pdfMap;
   std::map<std::string,std::unique_ptr<RooAbsData>> dataMap;

   RooArgSet observables;
   RooArgSet models;

   auto nChannels = 10;

   for(std::size_t i = 0; i < nChannels; ++i) {
       std::string suffix = "_" + std::to_string(i + 1);
       auto obsName = "x" + suffix;
       auto x = std::make_unique<RooRealVar>(obsName.c_str(), obsName.c_str(), 0, 10.);
       x->setBins(5);

       std::unique_ptr<RooAbsPdf> model{createModel(*x, std::to_string(i + 1))};

       pdfMap.insert({"channel" + suffix, model.get()});
       channelCat.defineType("channel" + suffix, i);
       dataMap.insert({"channel" + suffix, std::unique_ptr<RooAbsData>{model->generateBinned(*x, nEvents)}});
       //dataMap.insert({"channel" + suffix, std::unique_ptr<RooAbsData>{model->generate(*x, nEvents)}});

       observables.addOwned(std::move(x));
       models.addOwned(std::move(model));
   }


   RooSimultaneous model{"model", "model", pdfMap, channelCat};

   // Generate the same dataset for all backends.
   RooDataSet data{"data", "data", {observables, channelCat}, Index(channelCat), Import(dataMap)};

   // Save the original parameter values and errors to reset after minimization
   RooArgSet params;
   model.getParameters(&observables, params);

   RooFitADBenchmarksUtils::randomizeParameters(params, 1337);

   RooArgSet origParams;
   params.snapshot(origParams);

   std::unique_ptr<RooAbsReal> nllRef{model.createNLL(data, RooFit::BatchMode("off"))};
   std::unique_ptr<RooAbsReal> nllRefBatch{model.createNLL(data, RooFit::BatchMode("cpu"))};
   auto nllRefResolved = static_cast<RooAbsReal *>(nllRefBatch->servers()[0]);

   std::string name = "myNll" + std::to_string(counter);
   RooFuncWrapper nllFunc(name.c_str(), name.c_str(), *nllRefResolved, observables, &data, &model);
   //nllFunc.dumpCode();

   std::unique_ptr<RooMinimizer> m = nullptr;
   for (auto _ : state) {
      int code = state.range(0);
      if (code == RooFitADBenchmarksUtils::backend::Reference) {
         m.reset(new RooMinimizer(*nllRef));
      } else if (code == RooFitADBenchmarksUtils::backend::CodeSquashNumDiff) {
         m.reset(new RooMinimizer(nllFunc));
      } else if (code == RooFitADBenchmarksUtils::backend::BatchMode) {
         m.reset(new RooMinimizer(*nllRefBatch));
      } else if (code == RooFitADBenchmarksUtils::backend::CodeSquashAD) {
         RooMinimizer::Config minimizerCfgAd;
         minimizerCfgAd.gradFunc = [&](double *out) { nllFunc.getGradient(out); };
         m.reset(new RooMinimizer(nllFunc, minimizerCfgAd));
      }

      m->setPrintLevel(-1);
      //m->setPrintEvalErrors(-1);
      m->setStrategy(0);
      params.assign(origParams);
      m->minimize("Minuit2");
   }
   //m->save()->Print();
   std::cout << m->save()->status() << std::endl;
   std::cout << m->save()->minNll() << std::endl;
}

int main(int argc, char **argv)
{

   //RooFitADBenchmarksUtils::doBenchmarks(BM_RooFuncWrapper_Minimization, 10000, 10000, 10);
   RooFitADBenchmarksUtils::doBenchmarks(BM_RooFuncWrapper_ManyParams_Minimization, 1000, 1000, 10);

   benchmark::Initialize(&argc, argv);
   benchmark::RunSpecifiedBenchmarks();
   benchmark::Shutdown();
}
