# PSO-for-Case-Retrieval
Case retrieval model based on PSO algorithm.
  1. Similarity calculation<br>
    (1) Text-based data: Converting text into word frequency vectors using bag-of-words modeling<br>
    (2) Similarity matrix data: Taking values according to the similarity matrix<br>
    (3) Interval-type data: Calculated using the formula sim(a,b)=a⋂b/(a+b-a∩b)<br>
    (4) Float-type data: Calculated using the formula sim(a,b)=a⋂b/(a+b-a∩b)<br>
    (5) Categorical-type data: Calculated using the formula sim(a,b)=1-|a-b|/6<br>
    <br>
  2. Implementation of the PSO algorithm
     <br>
  3. Plotting fitness-number of iterations
