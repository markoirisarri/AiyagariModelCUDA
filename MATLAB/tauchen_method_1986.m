function [zgrid, Pi] = tauchen_method_1986(N, mu, rho, sigma, b)
    % Default bandwidth value if not provided
    if nargin < 5
        b = 3;
    end

    % Obtain sigma of the z's

    sigma_z = (sigma)/(sqrt(1-rho^2));

    % Discretize the AR(1) process using Tauchen method
    zgrid = linspace(mu - b * sigma_z, mu + b * sigma_z, N);
    dz = (zgrid(end) - zgrid(1)) / (N - 1);
    
    zgrid = zgrid(:);

    % Compute transition probabilities
    Pi = zeros(N, N);
    for i = 1:N
        for j = 1:N
            if j == 1
                Pi(i, j) = normcdf((zgrid(j) - rho * zgrid(i) + dz / 2) / sigma);
            elseif j == N
                Pi(i, j) = 1 - normcdf((zgrid(j) - rho * zgrid(i) - dz / 2) / sigma);
            else
                Pi(i, j) = normcdf((zgrid(j) - rho * zgrid(i) + dz / 2) / sigma) - ...
                            normcdf((zgrid(j) - rho * zgrid(i) - dz / 2) / sigma);
            end
        end
    end
end
