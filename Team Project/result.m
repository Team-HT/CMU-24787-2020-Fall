clear all;
clc;
h = waitbar(0,'Loading Data and Model Trained Net...');

load('TransferLearning');

load('ACTrain');
load('ACTest');
load('ACValidate');

h = waitbar(0.33,'Classifying for Model...');
close(h);
inputSize = netTransfer.Layers(1).InputSize;
augimdsTest = augmentedImageDatastore(inputSize(1:2),xValidate,'ColorPreprocessing','gray2rgb');
[YPModel,~] = classify(netTransfer,augimdsTest);
sum = 0;

for i = 1:length(yValidate)
    if (yValidate(i) == YPModel(i))
        sum = sum+1;
    end
end
ModelAccuracy = sum/length(yValidate)*100;
f = figure(1);
set(f,'visible','off');
figure(2);
CM1 = confusionchart(yValidate,YPModel);
CM1.ColumnSummary = 'column-normalized';
CM1.RowSummary = 'row-normalized';
CM1.Title = 'Demo Model Confusion Matrix';



