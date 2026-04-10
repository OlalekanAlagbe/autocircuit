---
name: neuronpedia-validate
description: This skill should be used when the user wants to validate data quality and ensure no mock data exists. Use when the user says "validate the data", "check data quality", or before publishing results.
---

# Neuronpedia Validate

Validate data quality and ensure no mock data in the pipeline.

## Instructions

### 1. Navigate to the pipeline directory

```bash
cd neuronpedia_pipeline
```

### 2. Run the validation script

```bash
python scripts/validate_real_data.py
```

Takes less than 1 second per file. Scans all JSON files in `data/graphs/` and checks multiple validation criteria.

### 3. Review the validation output

Note which files passed or failed, check the reasons for each status, and count real vs mock files.

### 4. Report the validation results

Include:
- Each file with its validation status
- For real data files: size, node count, confirmation
- For mock/invalid files: explain why they failed
- Summary: total real files vs mock files
- Overall status: Pass or Warning

## Validation Criteria

The script checks for:
1. **Filename prefix**: Must start with `real_`
2. **File size**: Must be >0.5 MB (real data is 2-6 MB)
3. **Circuit Tracer metadata**: Must have generator signature
4. **Node count**: Must have >100 nodes (real data has 500-1500)

## Output

**Console report** showing:
- Each file's validation status
- Reason for validation result
- Summary statistics
- Warning if mock data found

## Example Interaction

**User**: "Validate the data"

1. Navigate to neuronpedia_pipeline
2. Run validation script
3. Report:

```
Data validation complete:

Real data files: 4
  real_japan_currency.json (4.3 MB, 962 nodes)
  real_japan_currency_converted.json (2.7 MB, 858 nodes)
  real_france_capital.json (6.5 MB, 1088 nodes)
  real_france_capital_converted.json (4.0 MB, 985 nodes)

Mock/invalid files: 0
Status: All data confirmed as genuine Neuronpedia Circuit Tracer output
```

## Common Issues

- **"All files marked as mock"**: Files may need `real_` prefix -- rename them
- **"File too small"**: Graph download may have failed -- re-fetch with `/neuronpedia-fetch`
- **Mock data found**: Remove mock files or ensure they're not used in analysis

## When to Use

- User asks to "validate the data"
- Before starting analysis or visualization
- After fetching new data
- User questions data authenticity
- Before publishing results
- Quality control checkpoint

## Next Steps

After validation:
- **If passed**: Proceed with `/neuronpedia-analyze` or `/neuronpedia-visualize`
- **If failed**: Re-fetch problematic graphs or remove mock data

## Data Quality Policy

No mock data should be used unless explicitly specified. This skill enforces that policy by automatically detecting and flagging any mock or fabricated data.
