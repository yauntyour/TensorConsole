#pragma once
#include "common.hpp"

namespace tc::profiler
{
    // entries: [{prof_entry, count, ms, pct, phase}] collected from [prof] stdout
    J from_eng_prof(const std::vector<J> &entries, const J &phase_totals);
    // Chrome Trace Event Format (nsys / perfetto / tc_prof SDK)
    J from_chrome_trace(const J &trace);
    // Summary report (markdown + json) for CLI / panel download
    J make_report(const J &job, const std::vector<J> &metrics, const J &grad_order,
                  const J &last_grad, const J &profile, const std::string &log_tail);
}