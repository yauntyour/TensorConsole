#pragma once
#include "common.hpp"

namespace tc::adapter
{
    // Full task-parameter schema for the project kind (drives UI forms & CLI).
    J schema_for(const std::string &kind);

    // Grad oscilloscope manifest: output->input, expression left->right.
    // cfg = MetaModel _config.json (n_decoder / n_experts).
    J grad_order_template(const J &cfg);
    // Reorder observed parameter names against the template; unknowns appended.
    J order_observed(const J &tmpl, const std::vector<std::string> &names);

    // stdout parsers (MetaModel train)
    std::optional<J> parse_step_line(const std::string &line);   // ce/aux/fm/lr/|g|/tps/nodes
    std::optional<J> parse_pool_line(const std::string &line);   // pool used/reserved/peak/trim
    std::optional<J> parse_prof_line(const std::string &line);   // [prof] header/entry
    std::optional<J> parse_log_csv(const std::string &line);     // step,ce,aux,lr,grad_norm,unfrozen,sec

    // Build train argv from spec params against the schema.
    std::vector<std::string> train_argv(const J &params, const std::string &out);
}