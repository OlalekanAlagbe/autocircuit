# Feature Description Fetch Strategies

Three strategies are available when fetching feature descriptions from the Neuronpedia API during circuit analysis.

## Option 3: Smart Fetch (RECOMMENDED)

- Fetches top 10 features per supernode by activation
- ~90 features total (for typical 8-9 supernodes)
- Takes 30-60 seconds
- Provides enough data for accurate theme inference
- Best balance of speed and quality

## Option 1: Quick Fetch

- Fetches top 3 features per layer group
- ~15 features total
- Takes 10 seconds
- Minimal descriptions for quick exploration

## Option 2: Complete Fetch

- Fetches ALL features in the graph
- ~900 features total
- Takes 5-10 minutes
- Exhaustive coverage (rarely needed)

## Details

- **Description format**: "Activates on: token1, token2, token3"
- **Caching**: Descriptions are cached for performance on subsequent runs
- **Smart fetch targets**: Most activated features per supernode, which are most relevant for theme inference
- **Performance**: Smart fetch is 10x faster than Complete fetch with similar theme quality
