clearvars; % 可选，用于清除工作区，确保干净的开始
close all;
clc;
%% PPO Model Network and Code
%% Flags / Settings
parallelComputing_flag = 0;  % Whether use Parallel computing
load_Saved_Agent_flag = 0;
maxEpisodes = 10000;
maxSteps = 200;
dt = 0.05;
rng(0);
nL = 128;                            % number of neurons
%% Load Saved Agent
if load_Saved_Agent_flag == 1
    savedAgent_dir = 'saved_Agents01';   
    listing = dir(fullfile(savedAgent_dir, '*.mat'));
    for i = 1:length(listing)
         temp_String= string(listing(i).name);
         temp_String = extractAfter(temp_String,5); 
         temp_String = extractBefore(temp_String,'.mat'); 
         agent_names(i,1) = str2num(temp_String);
         
    end
    sorted_agent_names = sort(agent_names,'ascend');
    last_Agent = sorted_agent_names(end);
    agent_Name = append('\Agent',num2str(last_Agent), '.mat');
    load([savedAgent_dir agent_Name]);
    [ep_reward ep_no] = max(savedAgentResult.EpisodeReward);
    load([savedAgent_dir append('\Agent', num2str(ep_no), '.mat')]);
    plot(savedAgentResult)
end

rl_env = CartPoleEnv2();
rl_env.Ts = dt; % Ensure this matches the agent's SampleTime if set explicitly
               % Or better, use env.Ts for agent's SampleTime
T_Sample = rl_env.Ts;
% --- Get Observation and Action Information from the Environment ---
obsInfo = getObservationInfo(rl_env);
actInfo = getActionInfo(rl_env);

Act1_Min = actInfo.LowerLimit;
Act1_Max = actInfo.UpperLimit;
nI = obsInfo.Dimension(1);  % number of inputs
nO = actInfo.Dimension(1);    % number of outputs

criticNetwork = [
    featureInputLayer(nI,'Normalization','none','Name','observation')
    fullyConnectedLayer(nL,'Name','fc1')
    reluLayer('Name','relu1')
    fullyConnectedLayer(nL,'Name','fc2')
    reluLayer('Name','relu2')
    fullyConnectedLayer(nL,'Name','fc3')
    reluLayer('Name','relu3')
    fullyConnectedLayer(1,'Name','fc4')];

%% Critic Netwrok

criticNetwork = dlnetwork(criticNetwork);  
criticOptions = rlOptimizerOptions('Optimizer','adam','LearnRate',1e-4,'GradientThreshold',1,'L2RegularizationFactor',2e-4); %Use GPU for Training
critic = rlValueFunction(criticNetwork,obsInfo,'Observation',{'observation'},'UseDevice',"cpu");

commonPath = [
    featureInputLayer(nI,'Normalization','none','Name','comPathIn')
    fullyConnectedLayer(nL,'Name','fc1_c')
    reluLayer('Name','relu1_c')
    fullyConnectedLayer(nL,'Name','fc2_c')
    reluLayer('Name','comPathOut')
];

meanPath = [
    fullyConnectedLayer(1,'Name','meanPathIn')
    tanhLayer('Name','tanh1_m')
    scalingLayer('Name','meanPathOut','Scale',Act1_Max,'Bias',-0.5)
    ];

sdevPath = [
    fullyConnectedLayer(1,'Name','stdPathIn')
    softplusLayer('Name','stdPathOut')
    ];

actorNetwork = layerGraph(commonPath);
actorNetwork = addLayers(actorNetwork,meanPath);
actorNetwork = addLayers(actorNetwork,sdevPath);

actorNetwork = connectLayers(actorNetwork,"comPathOut","meanPathIn/in");
actorNetwork = connectLayers(actorNetwork,"comPathOut","stdPathIn/in");
actorOptions = rlOptimizerOptions('Optimizer','adam','LearnRate',5e-5,'GradientThreshold',1,'L2RegularizationFactor',1e-5);
actor = rlContinuousGaussianActor(actorNetwork,obsInfo,actInfo,'ActionMeanOutputNames',{'meanPathOut'}, ...
    'ActionStandardDeviationOutputNames',{'stdPathOut'},'ObservationInputNames',{'comPathIn'},'UseDevice','gpu'); %Use GPU for Training


% figure('Name','Actor Network');
% plot(actorNetwork);

%% Agent Options

agentOptions = rlPPOAgentOptions('SampleTime',T_Sample, ExperienceHorizon=1024,ClipFactor=0.2, ...
    EntropyLossWeight=0.05,NumEpoch=3,AdvantageEstimateMethod="gae",GAEFactor=0.5, ...
    DiscountFactor=0.997,ActorOptimizerOptions=actorOptions,CriticOptimizerOptions=criticOptions);

agent = rlPPOAgent(actor,critic,agentOptions);

%% Specify Training Options and Train Agent


% Configure Parallelization Options
parallelOptions = rl.option.ParallelTraining(...
    'Mode', 'async'); % Async Parallel Training Mode

% Define training options for the reinforcement learning agent
trainingOptions = rlTrainingOptions(...
    'MaxEpisodes', maxEpisodes, ...
    'MaxStepsPerEpisode', maxSteps, ...
    'ScoreAveragingWindowLength', 100, ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'StopTrainingCriteria', 'EpisodeCount', ...
    'StopTrainingValue', maxEpisodes, ...
    'SaveAgentCriteria', 'EpisodeSteps', ...
    'SaveAgentValue', 900, ...
    'SaveAgentDirectory', 'savedAgents_1', ...
    'UseParallel', false, ...
    'ParallelizationOptions', parallelOptions);

if parallelComputing_flag==1
    save_system(mdl);
    % Set up the parallel pool using 75% of available CPU cores
    num_cores = feature('numcores');
    parpool(floor(num_cores * 0.5));
    % Ensure the GPU is selected on all workers
    parfevalOnAll(@() gpuDevice(1), 0);
end

%% Train the agent.
trainingStats = train(agent,rl_env,trainingOptions)

%% Simulate PPO Agent
%可视化最后一个
animate_cartpole_motion_new(rl_env.EpisodeTrajectory,rl_env.DynParams)

simOptions = rlSimulationOptions('MaxSteps',maxSteps);
experience = sim(rl_env,agent,simOptions);

%再可视化一次
animate_cartpole_motion_new(rl_env.EpisodeTrajectory,rl_env.DynParams)

haha = true




