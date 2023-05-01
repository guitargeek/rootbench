#ifndef BenchmarkUtils_h
#define BenchmarkUtils_h

#include <RooArgSet.h>
#include <RooRandom.h>
#include <RooAbsRealLValue.h>

#include "benchmark/benchmark.h"

namespace RooFitADBenchmarksUtils {

enum backend { Reference, BatchMode, CodeSquashNumDiff, CodeSquashAD };
template <typename F>
void doBenchmarks(F func, int rangeMin, int rangeMax, int step, int numIterations = 1,
                  benchmark::TimeUnit unit = benchmark::kMillisecond)
{
   // Run the minimization with the reference NLL
   benchmark::RegisterBenchmark("NllReferenceMinimization", func)
      ->ArgsProduct({{Reference}, benchmark::CreateRange(rangeMin, rangeMax, step)})
      ->Unit(unit)
      ->Iterations(numIterations);

   // Run the minimization with the reference NLL (BatchMode)
   benchmark::RegisterBenchmark("NllBatchModeMinimization", func)
      ->ArgsProduct({{BatchMode}, benchmark::CreateRange(rangeMin, rangeMax, step)})
      ->Unit(unit)
      ->Iterations(numIterations);

   // Run the minimization with the code-squashed version with numerical-diff.
   benchmark::RegisterBenchmark("NllCodeSquash_NumDiff", func)
      ->ArgsProduct({{CodeSquashNumDiff}, benchmark::CreateRange(rangeMin, rangeMax, step)})
      ->Unit(unit)
      ->Iterations(numIterations);

   // Run the minimization with the code-squashed version with AD.
   benchmark::RegisterBenchmark("NllCodeSquash_AD", func)
      ->ArgsProduct({{CodeSquashAD}, benchmark::CreateRange(rangeMin, rangeMax, step)})
      ->Unit(unit)
      ->Iterations(numIterations);
}

void randomizeParameters(const RooArgSet &parameters, ULong_t seed = 0)
{
   auto random = RooRandom::randomGenerator();
   if (seed != 0)
      random->SetSeed(seed);

   for (auto param : parameters) {
      auto par = dynamic_cast<RooAbsRealLValue *>(param);
      if(!par) continue;
      const double uni = random->Uniform();
      const double min = par->getMin();
      const double max = par->getMax();
      par->setVal(min + uni * (max - min));
   }
}
} // namespace RooFitADBenchmarksUtils
#endif
