clear;
clc;
close all;
rng(0)

nSequences = 1000; % how many real sequences to put in the data

%% Specs
TF = [0,1,0,0,0,0,0,0;
      0,0,1,0,0,0,0,0;
      0,0,0,1,0,0,0,0;
      0,0,0,0,0,0,0,0;
      0,0,0,0,0,1,0,0;
      0,0,0,0,0,0,1,0;
      0,0,0,0,0,0,0,1;
      0,0,0,0,0,0,0,0]; % transition matrix
TR = TF';
nSensors = 273;
nTrainPerStim = 18; % training examples per stimulus
nNullExamples = nTrainPerStim * 8; % number of null examples
nSamples = 6000; % 60 seconds (at 100 Hz) of unlabelled data to predict
maxLag = 60; % evaluate time lag up to 600ms
cTime = 0:10:maxLag*10; % the millisecond values of each cross-correlation lag
nSubj = 24; % number of subjects to simulate
gamA = 10;
gamB = 0.5; % parameters for the gamma distribution
[~, pInds] = uperms(1:8,29);
uniquePerms = pInds;
nShuf = size(uniquePerms, 1);
samplerate = 100;
nstates = 8;

% Preallocate cell arrays to store, for each subject, 500 probabilities.
% For pre-injection (baseline), the target and non-target values are the same.
probaTarget_pre = cell(nSubj, 1);  % baseline probability samples from timepoints without injection.
probaNonTarget_pre = cell(nSubj, 1);  % same as probaTarget_pre.
% For post-injection we sample from the injection events:
% probaTarget_post gets the probability from the injected (target) class,
% probaNonTarget_post gets a probability from a non-injected (non-target) class.
probaTarget_post = cell(nSubj, 1);
probaNonTarget_post = cell(nSubj, 1);

numSamplesPerSubject = 500;

for iSj = 1:nSubj
    disp(['Processing Subject ' num2str(iSj) '...'])
    
    %% Generate sensor covariance structure
    A = randn(nSensors);
    [U, ~] = eig((A + A')/2);
    covMat = U * diag(abs(randn(nSensors, 1))) * U';
    
    %% Generate true patterns
    commonPattern = randn(1, nSensors);
    patterns = repmat(commonPattern, [8, 1]) + randn(8, nSensors);
    
    %% Create training data
    trainingData = 4 * randn(nNullExamples + 8 * nTrainPerStim, nSensors) + [zeros(nNullExamples, nSensors); repmat(patterns, [nTrainPerStim, 1])];
    trainingLabels = [zeros(nNullExamples, 1); repmat((1:8)', [nTrainPerStim, 1])];
    
    %% Add extra noise to selected patterns
    MoreNoiseind = randsample(1:8, 4);
    indend = MoreNoiseind * nTrainPerStim;
    indstart = (MoreNoiseind - 1) * nTrainPerStim + 1;
    nindex = [];
    for iind = 1:length(MoreNoiseind)
        xtemp = indstart(iind):indend(iind);
        nindex = [nindex, xtemp];
    end
    trainingData(nNullExamples + nindex, :) = trainingData(nNullExamples + nindex, :) + randn(length(nindex), nSensors);
    
    %% Train classifiers (one per stimulus) using lasso logistic regression
    betas = nan(nSensors, 8);
    intercepts = nan(1, 8);
    for iC = 1:8
        [betas(:, iC), fitInfo] = lassoglm(trainingData, trainingLabels == iC, 'binomial', 'Alpha', 1, 'Lambda', 0.006, 'Standardize', false);
        intercepts(iC) = fitInfo.Intercept;
    end
    
    %% Generate long unlabelled data with sensor dependence
    X = nan(nSamples, nSensors);
    X(1, :) = randn(1, nSensors);
    
    for iT = 2:nSamples
         X(iT, :) = 0.95 * (X(iT-1, :) + mvnrnd(zeros(1, nSensors), covMat));
    end

    X_pre =  X(:,:);

    
    %% Prepare arrays to record injection events (time and class) for this subject
    injectionTimes = [];
    injectionClasses = [];
    
    %% Inject sequences into X and record injection events
    for iRS = 1:nSequences
        seqTime = randi([40, nSamples - 40]); % choose a timepoint away from the edges
        state = false(8, 1);
        state(randi(8)) = true;  % randomly choose a starting state
        
        for iMv = 1:2  % two step replay
            if sum(state) == 0
                % Do nothing if state is empty
                X(seqTime, :) = X(seqTime, :);
            else
                % Record the injection event
                injectionTimes = [injectionTimes; seqTime];
                injectionClasses = [injectionClasses; find(state)];
                
                % Inject the pattern corresponding to the current state
                X(seqTime, :) = X(seqTime, :) + patterns(state, :);
                
                % Advance state via transition matrix with randomness
                state = (state' * TR)';
                state2 = false(8, 1);
                state2(find(rand < cumsum(state), 1, 'first')) = true;
                state = state2;
                
                % Move seqTime ahead by a gamma-randomized interval
                seqTime = seqTime + round(gamrnd(gamA, gamB));
            end
        end
    end
    
    %% Compute predictions using the trained classifiers
    preds = 1 ./ (1 + exp(-(X * betas + repmat(intercepts, [nSamples, 1]))));  % with replay
    preds_pre = 1 ./ (1 + exp(-(X_pre * betas + repmat(intercepts, [nSamples, 1]))));  % without replay
    
    % normalize the probabilities same as we do in the Python code
    % divide by mean of class

    %preds2 = preds ./ mean(preds);
    %preds_pre2 = preds_pre ./ mean(preds_pre);

    %% --- Extract Probabilities ---
    % Here we use the injection events:
    % For the target condition, we extract the probability from the injected class.
    % For the non-target, at the same injection timepoints we extract a probability
    % from a randomly chosen classifier (which is not the injected class).
    preProbs_target = zeros(numSamplesPerSubject, 1);
    preProbs_nontarget = zeros(numSamplesPerSubject, 1);
    postProbs_target = zeros(numSamplesPerSubject, 1);
    postProbs_nontarget = zeros(numSamplesPerSubject, 1);

    for ii = 1:numSamplesPerSubject
       postProbs_target(ii) = preds(injectionTimes(ii), injectionClasses(ii));
       preProbs_target(ii) = preds_pre(injectionTimes(ii), injectionClasses(ii));

       candidateClasses = setdiff(1:8, injectionClasses(ii));
       chosenIdx = candidateClasses(randi(length(candidateClasses)));
       postProbs_nontarget(ii) = preds(injectionTimes(ii), chosenIdx);
       preProbs_nontarget(ii) = preds_pre(injectionTimes(ii), chosenIdx);

    end
    probaTarget_pre{iSj} =  preProbs_target;
    probaNonTarget_pre{iSj} =  preProbs_nontarget;
    probaTarget_post{iSj} =  postProbs_target;
    probaNonTarget_post{iSj} =  postProbs_nontarget;

end

% Save the variables to a .mat file.
save('probabilities.mat', 'probaTarget_pre', 'probaNonTarget_pre', 'probaTarget_post', 'probaNonTarget_post', '-v7.3')

% At this point, for each subject the cell arrays probaTarget_pre and probaNonTarget_pre
% contain 500 baseline (pre-injection) probabilities, and the cell arrays
% probaTarget_post and probaNonTarget_post contain (up to) 500 probabilities extracted from
% injection events (target: the injected class; non-target: a different class).
