% Aiyagari Model (VFI component) Sample Code, vectorized version in Matlab 
% Author: Marko Irisarri

clear; close; clc;

% Define parameters

p.beta = 0.96;
p.gamma = 3;
p.rho = 0.9;
p.sigma = 0.08717;
p.dim_a = 100;
p.dim_y = 7;
p.min_a = 0.01;
p.max_a = 30;
p.alpha = 0.36;
p.delta = 0.08;

% Get AR(1) discretized process by Tauchen (1986)

[p.y_grid, p.Pi_y] = tauchen_method_1986(p.dim_y, 0, p.rho, p.sigma);

% Obtain (sparse) global transition matrix

p.global_transition = sparse(kron(sparse(p.Pi_y), speye(p.dim_a)));

% Get grid for assets

p.a_grid = linspace(p.min_a, p.max_a, p.dim_a);

% Get mesh

[p.A, p.Y] = ndgrid(p.a_grid, p.y_grid);

% Get index mesh

[p.index_a, p.index_y] = ndgrid(1:p.dim_a, 1:p.dim_y);

% Linearize grid

p.A = p.A(:);
p.Y = p.Y(:);

% Define utility function 

p.util = @(x) (max(x,0).^(1-p.gamma))./(1-p.gamma);

% Define parameters for VFI convergence (these are fixed for 
% comparign the running times with the CUDA counterpart)

tol_VFI = 1e-5;
tol_golden_section = 1e-5;
max_iter_VFI = 1000;

iter_VFI = 0;
diff_VFI = 1;

% Preallocate memory upper bound control

upper_bound = zeros(p.dim_a*p.dim_y,1);

% VFI block

p.samples_vfi = 100;

% Fix values for r and w 

r = 0.02;
w = 2;

tic;

for i = 1:p.samples_vfi

% Reinitialize VFI

i

value = zeros(p.dim_a * p.dim_y,1);
cont_value = value;
interpolant_value = griddedInterpolant(reshape(p.A, [p.dim_a, p.dim_y]), reshape(p.Y, [p.dim_a, p.dim_y]), reshape(cont_value, [p.dim_a p.dim_y]), 'linear', 'nearest');
value_old = value;
iter_VFI = 0;

while(iter_VFI < max_iter_VFI) 
  
  if(iter_VFI <= 15 || mod(iter_VFI, 15) == 0) 
  
  % Update policy and value functions
  
    f = @(x) p.util((1+r)*p.A + w*exp(p.Y) - x) + p.beta*interpolant_value(x,p.Y);
    
    upper_bound = (1+r).*p.A + w*exp(p.Y);
    
    [policy, value] = golden_section_search(f, p.min_a.*ones(p.dim_a*p.dim_y,1), upper_bound,tol_golden_section);
    
    cont_value = p.global_transition*value;
    
    interpolant_value = griddedInterpolant(reshape(p.A, [p.dim_a, p.dim_y]), reshape(p.Y, [p.dim_a, p.dim_y]), reshape(cont_value, [p.dim_a p.dim_y]), 'linear', 'nearest');
  
  else 
  
    % Howard Improvement, update value function with current policy function
    
    f = @(x) p.util((1+r)*p.A + w*exp(p.Y) - x) + p.beta*interpolant_value(x,p.Y);
     
    value = f(policy);
    
    cont_value = p.global_transition*value;
    
    interpolant_value = griddedInterpolant(reshape(p.A, [p.dim_a, p.dim_y]), reshape(p.Y, [p.dim_a, p.dim_y]), reshape(cont_value, [p.dim_a p.dim_y]), 'linear', 'nearest');
    
    end
    
    diff_VFI = norm(value - value_old);
    
    value_old = value;
    
    iter_VFI = iter_VFI + 1;
  
  end

end

toc;








