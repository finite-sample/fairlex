## Leximin Matching: Minimizing Maximum Covariate Distance

The Hungarian algorithm minimizes total covariate distance between treated and control units, ensuring optimal aggregate match quality. But this can leave some treated units with particularly poor covariate matches. Leximin matching offers an alternative objective that minimizes the maximum covariate distance first, then the second-maximum, and so on.

## Distance Measurement and Objective Functions

We measure covariate distance using Euclidean distance in standardized covariate space. For treated unit i with covariates X_i = [age_i, income_i, education_i] and control unit j with X_j = [age_j, income_j, education_j], the distance is:

d(i,j) = √[(age_i - age_j)² + (income_i - income_j)² + (education_i - education_j)²]

where covariates are standardized to have mean 0 and variance 1.

**Hungarian:** Minimize ∑ᵢ d(i, matched_j) over all valid assignments

**Leximin:** Minimize max{d(i, matched_j)} first, then second-max, then third-max, etc.

## When Hungarian and Leximin Disagree

Consider 3 treated units, 3 controls, with two possible complete assignments:

**Assignment A:** distances [1.0, 1.0, 9.0] → total = 11.0, max = 9.0  
**Assignment B:** distances [2.0, 3.0, 7.0] → total = 12.0, max = 7.0

- **Hungarian chooses A** (minimizes total: 11.0 < 12.0)
- **Leximin chooses B** (minimizes max: 7.0 < 9.0)

Hungarian accepts the terrible 9.0-distance match because it reduces total cost. Leximin rejects it to avoid leaving any unit with an extremely poor match. "Lexicographically" means optimizing distances in worst-to-best order: first minimize the worst distance, then among solutions achieving that minimum, minimize the second-worst, and so on.

## Practical Application: Citizens' Assembly Selection

A recent Nature paper applied leximin to selecting representative panels for democratic participation. Researchers developed algorithms that maximize the minimum probability any individual gets selected for a citizens' assembly, while maintaining demographic quotas. Their LEXIMIN algorithm has been deployed by organizations across multiple countries, selecting over 40 assemblies.

The parallel to matching is direct: ensure representativeness without systematically excluding any demographic combinations - analogous to achieving covariate balance without leaving treated units with extremely poor matches.

## Empirical Evidence

We compared Hungarian and leximin across three covariate distribution scenarios:

**Balanced distributions:** Both methods achieved similar maximum distances (2.0027) and treatment effect estimates. Method choice made minimal difference.

**Clustered distributions:** Leximin reduced maximum covariate distance by 19.6% (2.05 → 1.64) compared to Hungarian, though with slightly worse average balance.

**Sparse distributions with outliers:** Leximin achieved better treatment effect estimation (bias: -0.014 vs 0.330) despite having worse average covariate balance.

The consistent pattern: leximin sacrifices average balance to protect worst-matched units, which can improve causal inference when covariate distributions are challenging.

## Implementation

The computational challenge is that naive LP formulations with fractional variables produce incorrect results. Partial assignments artificially reduce maximum distance constraints. Solutions include:

1. **Integer programming** (computationally expensive)
2. **Bottleneck assignment** (reformulate as minimum bottleneck matching)

We use bottleneck assignment, which finds the minimum possible maximum edge weight in a perfect matching - equivalent to leximin for most practical problems while remaining computationally tractable.

## When to Use Leximin

**Choose leximin when:**
- Covariate distributions are clustered or contain outliers
- Worst-case matches pose analytical risks
- Individual match quality matters more than aggregate efficiency

**Choose Hungarian when:**
- Covariates are well-distributed
- Computational speed is critical
- Theoretical optimality guarantees are required

## Limitations

Results come from specific simulation settings. The 19.6% maximum distance improvement and better bias performance need validation across broader contexts. The relationship between individual fairness and causal inference quality requires deeper investigation.

Extension to 1-to-k matching adds complexity. Our sequential approach provides a reasonable heuristic, but optimal k-matching under leximin criteria remains algorithmically challenging.

## Conclusion

Leximin matching minimizes maximum covariate distance rather than total distance. In challenging scenarios with clustered covariates or outlier units, protecting worst-matched individuals can improve treatment effect estimation even when average covariate balance appears worse.

The choice between Hungarian and leximin reflects competing statistical objectives: aggregate optimality versus individual match quality. Both have their place depending on the covariate distribution and research priorities.
