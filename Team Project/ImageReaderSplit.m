% Example of using a datastore, see 
clear all;
clc;


rootdir = 'C:\Users\Yuqis\Desktop\Project';


Images = imageDatastore(...
    rootdir, ...
    'IncludeSubfolders',true, ...
    'LabelSource', 'foldernames');

rng('default')
rng(17)

[validateImages,testImages,trainImages] = splitEachLabel(Images,0.15,0.25,'randomized');

% fprintf('Read images into datastores\n');
% 
xTrain = trainImages;
yTrain = trainImages.Labels;
xTest = testImages;
yTest = testImages.Labels;
xValidate = validateImages;
yValidate = validateImages.Labels;

% 
% 
save('ACTrain.mat', 'xTrain', 'yTrain') 
save('ACTest.mat', 'xTest', 'yTest') 
save('ACValidate.mat', 'xValidate', 'yValidate') 
% % save('featuresVarify.mat', 'xVarify', 'yVarify') 




