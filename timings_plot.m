% Timings plot

clear; close all; clc;

assets_dimension = [100, 1000, 10000, 100000];
cuda_vfi_seconds = [0.545923 0.591243 0.880553 1.07062];
matlab_vfi_seconds = [47.63753 197.815513 1555.780887 1930.230246];

speed_ups = [matlab_vfi_seconds ./ cuda_vfi_seconds];

% Convert assets_dimension to categorical
assets_category = categorical({'100', '1000', '10000', '100000'}, ...
    'Ordinal', true);

% Create a bar plot with logarithmic y-axis
figure;
b = bar(assets_category, [cuda_vfi_seconds; matlab_vfi_seconds; speed_ups]', 'grouped');
set(gca, 'YScale', 'log');  % Set logarithmic scale on the y-axis

% Labeling and title
xlabel('Assets Dimension', 'Interpreter', 'latex', 'FontSize', 20);

ax = gca;
ax.XAxis.TickLabelInterpreter = 'latex';
ax.XAxis.FontSize = 20;

ylabel('Seconds (1-2) / Speed-up (3) (log scale)', 'Interpreter', 'latex', 'FontSize', 20);
title('CUDA (12.3) and MATLAB (2021b) Execution Time Comparison (VFI)', 'Interpreter', 'latex', 'FontSize', 22);
subtitle('Total Execution Time for 100 runs (10 runs when 100,000)', 'Interpreter', 'latex', 'FontSize', 20)

% Show legend
legend('CUDA', 'MATLAB', 'Speed-Up', 'Interpreter', 'latex', 'Location', 'northwest', 'FontSize', 16);

grid on;

% Display the values on top of each bar
for i = 1:numel(b)
    yval = b(i).YData;
    xval = b(i).XEndPoints;
    if i <= 2
        text(xval, yval, strcat(num2str(yval', '%.1f'), 's'), 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center', 'FontSize', 14);
    else
        text(xval, yval + 0.05*yval, strcat(num2str(yval', '%.1f'), 'x'), 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center', 'FontSize', 14);
    end
end