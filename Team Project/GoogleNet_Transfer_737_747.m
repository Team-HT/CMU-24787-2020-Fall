close all;

rootdir = 'C:\Users\Yuqis\OneDrive\Desktop\FinalProject/';
subdir = [rootdir 'train'];
trainImages = imageDatastore(...
    subdir, ...
    'IncludeSubfolders',true, ...
    'LabelSource', 'foldernames');
xTrain = imageDatastoreReader(trainImages);
yTrain = trainImages.Labels;

rootdir = 'C:\Users\Yuqis\OneDrive\Desktop\FinalProject';
subdir = [rootdir 'validate'];
valImages = imageDatastore(...
    subdir, ...
    'IncludeSubfolders',true, ...
    'LabelSource', 'foldernames');
xVal = imageDatastoreReader(valImages);
yVal = valImages.Labels;

rootdir = 'C:\Users\Yuqis\OneDrive\Desktop\FinalProject';
subdir = [rootdir 'test'];
testImages = imageDatastore(...
    subdir, ...
    'IncludeSubfolders',true, ...
    'LabelSource', 'foldernames');
yTest = testImages.Labels;

net = googlenet;

net.Layers(1);

inputSize = net.Layers(1).InputSize;

lgraph = layerGraph(net);

lgraph = removeLayers(lgraph, {'loss3-classifier','prob','output'});

numClasses = numel(categories(yTrain));
newLayers = [
    fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',10,'BiasLearnRateFactor',10)
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')];
lgraph = addLayers(lgraph,newLayers);

lgraph = connectLayers(lgraph,'pool5-drop_7x7_s1','fc');

layers = lgraph.Layers;
connections = lgraph.Connections;

layers(1:110) = freezeWeights(layers(1:110));
lgraph = createLgraphUsingConnections(layers,connections);



pixelRange = [-30 30];
scaleRange = [0.9 1.1];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange, ...
    'RandXScale',scaleRange, ...
    'RandYScale',scaleRange);

augimdsTrain = augmentedImageDatastore(inputSize(1:2),trainImages, ...
    'DataAugmentation',imageAugmenter);

augimdsValidation = augmentedImageDatastore(inputSize(1:2),valImages);

miniBatchSize = 10;
options = trainingOptions('sgdm', ...
    'ExecutionEnvironment','auto',...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',3, ...
    'InitialLearnRate',3e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',3, ...
    'Verbose',false, ...
    'Plots','training-progress');

netTransfer = trainNetwork(augimdsTrain,lgraph,options);

augimdsTest = augmentedImageDatastore(inputSize(1:2),testImages);
[yOutTrans,scoreTrans] = classify(netTransfer,augimdsTest);
C_trans = confusionmat(yOutTrans,yTest);
accuracy_trans = (C(1)+C(4))/length(yTest);
