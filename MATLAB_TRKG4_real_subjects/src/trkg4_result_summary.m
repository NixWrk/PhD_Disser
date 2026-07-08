function trkg4_result_summary(result)
%TRKG4_RESULT_SUMMARY Print compact result summary.

fprintf('\nTRKG4 result for %s\n', result.cfg.subject.id);
fprintf('  Z_model = %.6f Ohm\n', result.Z_model_ohm);

fprintf('\nTissues:\n');
for k = 1:numel(result.tissue_names)
    fprintf('  %-14s %8d elems\n', result.tissue_names{k}, result.tissue_counts(k));
end
end
