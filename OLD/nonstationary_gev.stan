data {
  int<lower=1> N;
  vector[N] y;
  vector[N] x;
}
parameters {
  real beta0;
  real beta1;
  real<lower=0> sigma;
  real<lower=-0.5, upper=0.5> xi;
}
model { 
  vector[N] mu;
  for (n in 1:N)
    mu[n] = beta0 + beta1 * x[n];

  // Priors
  beta0 ~ normal(0, 100);
  beta1 ~ normal(-100,100);    // uninformative prior
  sigma ~ normal(0, 50);
  xi ~ uniform(-0.5, 0.5);

  // Likelihood
  for (n in 1:N) {
    real z = 1 + xi * (y[n] - mu[n]) / sigma;
    if (z <= 0)
      target += negative_infinity();
    else
      target += -log(sigma) - (1 + 1/xi) * log(z) - pow(z, -1/xi);
  }
}

