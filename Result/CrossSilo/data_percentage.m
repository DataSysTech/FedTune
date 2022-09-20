% data percentage

clear, clc

ratio_arr = 10:10:100;
accuracy_mean = zeros(size(ratio_arr));
accuracy_std = zeros(size(ratio_arr));

for i_ratio = 1:length(ratio_arr)
   ratio = ratio_arr(i_ratio); 
   ratio_str = strrep(num2str(ratio / 100, '%.2f'), '.', '_');
   data_filename = strcat('data_percentage_', ratio_str, '.csv');
   data = dlmread(data_filename);
   epoch = data(:, 1);
   model_accuracy = data(:, 2:end);
   last_model_accuracy = model_accuracy(end, :);
   
   accuracy_mean(i_ratio) = mean(last_model_accuracy);
   accuracy_std(i_ratio) = std(last_model_accuracy);
end

figure(1), clf, hold on
set(gcf, 'position', [500, 500, 1000, 600])
bar(ratio_arr, accuracy_mean)

% error bar if you need
% er = errorbar(ratio_arr, accuracy_mean, accuracy_std);    
% er.Color = [0 0 0];                            
% er.LineStyle = 'none';
% er.LineWidth = 2;

p = polyfit(ratio_arr, accuracy_mean, 2);
y_fitted = polyval(p, ratio_arr);
plot(ratio_arr, y_fitted, 'linewidth', 2)

xlabel('Data Percentage (%)')
ylim([0.89, 0.93])
yticks(0.89:0.01:0.93)
ylabel('Model Accuracy')
set(gca, 'fontsize', 24, 'ygrid', 'on')

legend({'Mean accuracy', 'Curve fitting'}, 'location', 'northwest')
hold off
saveas(gcf, 'data_size_vs_model_accuracy.png')
