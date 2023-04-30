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

static void BM_RooFuncWrapper_ManyParams_Minimization(benchmark::State &state)
{

   counter++;

   // gInterpreter->ProcessLine("gErrorIgnoreLevel = 2001;");
   // auto &msg = RooMsgService::instance();
   // msg.setGlobalKillBelow(RooFit::WARNING);

   RooRealVar x("x", "x", 0, 10);
   RooRealVar c("c", "c", 0.1, 0, 10);

   RooExponential expo("expo", "expo", x, c);

   // Create two Gaussian PDFs g1(x,mean1,sigma) anf g2(x,mean2,sigma) and their parameters
   RooRealVar mean("mean", "mean of gaussians", 5, -10, 10);
   RooRealVar sigma1("sigma1", "width of gaussians", 0.50, .01, 10);
   RooRealVar sigma2("sigma2", "width of gaussians", 5, .01, 10);

   RooGaussian sig1("sig1", "Signal component 1", x, mean, sigma1);
   RooGaussian sig2("sig2", "Signal component 2", x, mean, sigma2);

   // Build Chebychev polynomial pdf
   RooRealVar a0("a0", "a0", 0.3, 0., 0.5);
   RooRealVar a1("a1", "a1", 0.2, 0., 0.5);
   RooChebychev bkg("bkg", "Background", x, RooArgSet(a0, a1));

   // Sum the signal components into a composite signal pdf
   RooRealVar sig1frac("sig1frac", "fraction of component 1 in signal", 0.8, 0., 0.9);
   RooAddPdf sig("sig", "Signal", RooArgList(sig1, sig2), sig1frac);

   // Sum the composite signal and background
   RooRealVar bkgfrac("bkgfrac", "fraction of background", 0.2, 0., 0.3);
   RooRealVar sigfrac("sigfrac", "fraction of signal", 0.2, 0., 0.3);
   RooRealVar expfrac("expfrac", "fraction of exponent", 0.26, 0., 0.3);
   RooAddPdf model("model", "g1+g2+a", RooArgList(bkg, sig, expo), {bkgfrac, sigfrac, expfrac});

   RooArgSet normSet{x};

   // Generate the same dataset for all backends.
   RooRandom::randomGenerator()->SetSeed(100);

   std::size_t nEvents = state.range(1);
   std::unique_ptr<RooDataSet> data0{model.generate(x, nEvents)};
   std::unique_ptr<RooAbsData> data{data0->binnedClone()};

   // Save the original parameter values and errors to reset after minimization
   RooArgSet params;
   model.getParameters(&normSet, params);

   RooFitADBenchmarksUtils::randomizeParameters(params, 1337);

   RooArgSet origParams;
   params.snapshot(origParams);

   std::unique_ptr<RooAbsReal> nllRef{model.createNLL(*data, RooFit::BatchMode("off"))};
   std::unique_ptr<RooAbsReal> nllRefBatch{model.createNLL(*data, RooFit::BatchMode("cpu"))};
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

int main(int argc, char **argv)
{

   RooFitADBenchmarksUtils::doBenchmarks(BM_RooFuncWrapper_Minimization, 10000, 10000, 10);
   RooFitADBenchmarksUtils::doBenchmarks(BM_RooFuncWrapper_ManyParams_Minimization, 10000, 10000, 10);

   benchmark::Initialize(&argc, argv);
   benchmark::RunSpecifiedBenchmarks();
   benchmark::Shutdown();
}
