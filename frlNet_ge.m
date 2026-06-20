% frlnet_policy_manager_ge_sim.m
% FRLNet Policy Manager simulation with Gilbert-Elliott wireless channels.
%
% Features:
%   - Discrete-time two-queue system
%   - Backlog vector V(t) = [Q1(t); Q2(t)]
%   - Aggregate backlog objective sum_i Q_i
%   - Event-driven episodes
%   - Exploration phase: model-free sampling
%   - Exploitation phase: model-based value iteration on estimated truncated MDP
%   - Hidden Gilbert-Elliott wireless channel
%   - No central server; decisions are made by the policy manager logic
%
% Notes:
%   - The scheduler does NOT observe channel states directly.
%   - The channel only affects observed service success/failure.
%   - This script is a clean simulation use-case for the paper.

clear; close all; clc;
rng(1); % reproducibility

% -------------------------------------------------------------------------
% 1) Simulation parameters
% -------------------------------------------------------------------------

% Queue arrival rates (packets/slot)
lambda = [0.35, 0.25];

% Gilbert-Elliott channel parameters
% channelState(i) = 1 -> Good state for link i
% channelState(i) = 0 -> Bad  state for link i
piGood = [0.90, 0.90];   % success probability in Good state
piBad  = [0.55, 0.55];   % success probability in Bad state
pGB = 0.05;              % P(G -> B)
pBG = 0.15;              % P(B -> G)

% Queue truncation threshold (packets)
T = 10;

% Practical low-backlog inner region approximation
gammaBound = 1;
T_in = max(T - gammaBound, 0);

% Policy-manager hyperparameters
r = 0.5;                 % exploration decay parameter
R = 10;                  % episode-transition threshold parameter
discountFactor = 0.98;   % discount factor for value iteration
valueTol = 1e-4;         % convergence tolerance for value iteration
maxVIters = 500;         % maximum VI iterations

% Simulation horizon and episode limit
totalSlotsTarget = 10000;
maxEpisodes = 500;

% -------------------------------------------------------------------------
% 2) State space and storage
% -------------------------------------------------------------------------

% Two-queue truncated state space: q1,q2 in {0,...,T}
numStates = (T + 1)^2;
numActions = 2; % action 1 -> serve queue 1, action 2 -> serve queue 2

% Transition counts for empirical MDP estimation
% transCounts(nextState, currentState, action)
transCounts = ones(numStates, numStates, numActions);

% Traces
backlogTrace        = zeros(totalSlotsTarget, 1);
queue1Trace         = zeros(totalSlotsTarget, 1);
queue2Trace         = zeros(totalSlotsTarget, 1);
episodeTrace        = zeros(totalSlotsTarget, 1);
phaseTrace          = zeros(totalSlotsTarget, 1); % 1 = explore, 2 = exploit
actionTrace         = zeros(totalSlotsTarget, 1);
channelStateTrace   = zeros(totalSlotsTarget, 2);
serviceSuccessTrace = zeros(totalSlotsTarget, 1);

% Initial backlog state V(t)
V = [0; 0];

% Initial channel state: start both links in Good state
channelState = [1; 1];

% Episode counter theta
episode = 1;

% Time-slot counter
timeSlot = 0;

% Episode length log
episodeLengths = [];

% -------------------------------------------------------------------------
% 3) Main simulation loop
% -------------------------------------------------------------------------
while timeSlot < totalSlotsTarget && episode <= maxEpisodes

    % Episode-level random variable mu in [0,1]
    mu = rand;

    % Episode-dependent thresholds
    zeta_theta = r / sqrt(episode);
    R_theta    = R * sqrt(episode);

    % Decide whether this episode is exploration or exploitation
    isExplorationEpisode = (mu <= zeta_theta);

    % Estimate the truncated MDP from all observed transitions so far
    P_hat = estimateTransitionModel(transCounts);

    % Cost is aggregate backlog: C_f(V) = Q1 + Q2
    costVec = buildCostVector(T);

    % Solve estimated truncated MDP via value iteration
    [~, policyStar] = valueIterationMDP( ...
        P_hat, costVec, discountFactor, valueTol, maxVIters);

    % Episode-local counters
    episodeSlotCounter = 0;
    episodeVisitsIn    = 0;
    episodeClosed      = false;

    % -------------------------------------------------------------
    % Run one event-driven episode
    % -------------------------------------------------------------
    while timeSlot < totalSlotsTarget

        timeSlot = timeSlot + 1;
        episodeSlotCounter = episodeSlotCounter + 1;

        % Current backlog state
        q1 = V(1);
        q2 = V(2);

        % Map state to truncated MDP index
        sIdx = state2idx(q1, q2, T);

        % Vmax(t) = max_i V_i(t)
        Vmax = max(V);

        % Practical approximation of S_in
        isIn = (Vmax <= T_in);

        % ---------------------------------------------------------
        % Action selection
        % ---------------------------------------------------------
        if isExplorationEpisode
            % Model-free exploration:
            % random action inside S_in, baseline policy outside S_in
            if isIn
                action = randi(numActions);
            else
                action = baselinePolicy(V);
            end
            phase = 1;
        else
            % Model-based exploitation:
            % use learned policy inside S_in, baseline policy outside S_in
            if isIn
                action = policyStar(sIdx);
            else
                action = baselinePolicy(V);
            end
            phase = 2;
        end

        % ---------------------------------------------------------
        % One-step system evolution with Gilbert-Elliott channels
        % ---------------------------------------------------------
        [Vnext, channelState, successFlag] = simulateQueueStepGE( ...
            V, action, lambda, channelState, piGood, piBad, pGB, pBG, T);

        % Update transition counts for empirical model learning
        nextIdx = state2idx(Vnext(1), Vnext(2), T);
        transCounts(nextIdx, sIdx, action) = transCounts(nextIdx, sIdx, action) + 1;

        % Log traces
        backlogTrace(timeSlot)        = sum(V);
        queue1Trace(timeSlot)         = V(1);
        queue2Trace(timeSlot)         = V(2);
        episodeTrace(timeSlot)        = episode;
        phaseTrace(timeSlot)          = phase;
        actionTrace(timeSlot)         = action;
        channelStateTrace(timeSlot,:)  = channelState.';
        serviceSuccessTrace(timeSlot)  = successFlag;

        % Count visits inside the low-backlog region
        if isIn
            episodeVisitsIn = episodeVisitsIn + 1;
        end

        % Advance to next state
        V = Vnext;

        % ---------------------------------------------------------
        % Episode termination condition
        % ---------------------------------------------------------
        % The episode ends once the threshold R_theta is reached.
        if episodeVisitsIn >= R_theta
            episodeLengths(end + 1) = episodeSlotCounter; %#ok<SAGROW>
            episodeClosed = true;
            break;
        end
    end

    % If we hit the simulation horizon before closing the episode,
    % still record the partial episode length.
    if ~episodeClosed && episodeSlotCounter > 0
        episodeLengths(end + 1) = episodeSlotCounter; %#ok<SAGROW>
    end

    % Move to next episode
    episode = episode + 1;
end

% Trim traces to actual simulated length
validIdx = 1:timeSlot;
backlogTrace       = backlogTrace(validIdx);
queue1Trace        = queue1Trace(validIdx);
queue2Trace        = queue2Trace(validIdx);
episodeTrace       = episodeTrace(validIdx);
phaseTrace         = phaseTrace(validIdx);
actionTrace        = actionTrace(validIdx);
channelStateTrace  = channelStateTrace(validIdx,:);
serviceSuccessTrace= serviceSuccessTrace(validIdx);

% -------------------------------------------------------------------------
% 4) Plots
% -------------------------------------------------------------------------

% Aggregate backlog plot
figure;
plot(validIdx, backlogTrace, 'LineWidth', 1.8);
xlabel('Time slot', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('Aggregate backlog \Sigma_i Q_i', 'FontSize', 14, 'FontWeight', 'bold');
title('FRLNet Policy Manager: Aggregate Backlog');
grid on;
ax = gca;
ax.FontSize = 12;
ax.FontWeight = 'bold';
set(gca, 'LooseInset', max(get(gca, 'TightInset'), 0.02));
exportgraphics(gcf, 'frlnet_ge_backlog_trace.png', 'Resolution', 600);

% Queue backlog plot
figure;
plot(validIdx, queue1Trace, 'LineWidth', 1.6); hold on;
plot(validIdx, queue2Trace, 'LineWidth', 1.6);
xlabel('Time slot', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('Queue backlog', 'FontSize', 14, 'FontWeight', 'bold');
legend({'Q_1(t)', 'Q_2(t)'}, 'Location', 'best', 'FontSize', 12);
title('Per-Queue Backlog');
grid on;
ax = gca;
ax.FontSize = 12;
ax.FontWeight = 'bold';
set(gca, 'LooseInset', max(get(gca, 'TightInset'), 0.02));
exportgraphics(gcf, 'frlnet_ge_queue_backlog.png', 'Resolution', 600);

% Channel-state plot
figure;
stairs(validIdx, channelStateTrace(:,1), 'LineWidth', 1.5); hold on;
stairs(validIdx, channelStateTrace(:,2), 'LineWidth', 1.5);
ylim([-0.2 1.2]);
xlabel('Time slot', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('Channel state', 'FontSize', 14, 'FontWeight', 'bold');
legend({'Link 1', 'Link 2'}, 'Location', 'best', 'FontSize', 12);
title('Gilbert-Elliott Channel States');
grid on;
ax = gca;
ax.FontSize = 12;
ax.FontWeight = 'bold';
set(gca, 'LooseInset', max(get(gca, 'TightInset'), 0.02));
exportgraphics(gcf, 'frlnet_ge_channels.png', 'Resolution', 600);

% Episode length plot
figure;
plot(episodeLengths, '-o', 'LineWidth', 1.8);
xlabel('Episode index', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('Episode length (slots)', 'FontSize', 14, 'FontWeight', 'bold');
title('Event-Driven Episode Lengths');
grid on;
ax = gca;
ax.FontSize = 12;
ax.FontWeight = 'bold';
set(gca, 'LooseInset', max(get(gca, 'TightInset'), 0.02));
exportgraphics(gcf, 'frlnet_ge_episode_lengths.png', 'Resolution', 600);

% -------------------------------------------------------------------------
% 5) Console summary
% -------------------------------------------------------------------------
fprintf('Simulation finished.\n');
fprintf('  Total slots simulated : %d\n', timeSlot);
fprintf('  Episodes completed    : %d\n', episode - 1);
fprintf('  Mean aggregate backlog: %.3f packets\n', mean(backlogTrace));
fprintf('  Final state           : [Q1, Q2] = [%d, %d]\n', V(1), V(2));

% =========================================================================
% Local helper functions
% =========================================================================

function idx = state2idx(q1, q2, T)
    % Map (q1,q2) in {0,...,T}^2 to a unique index in 1...(T+1)^2
    q1 = min(max(round(q1), 0), T);
    q2 = min(max(round(q2), 0), T);
    idx = sub2ind([T + 1, T + 1], q1 + 1, q2 + 1);
end

function [Vnext, channelState, successFlag] = simulateQueueStepGE( ...
        V, action, lambda, channelState, piGood, piBad, pGB, pBG, T)
    % One-step queue evolution under Gilbert-Elliott wireless channels.
    %
    % The policy manager does NOT directly observe channelState.
    % It only sees the resulting transition behavior.

    q1 = V(1);
    q2 = V(2);
    successFlag = 0;

    % -------------------------------------------------------------
    % 1) Channel evolution (hidden from the scheduler)
    % -------------------------------------------------------------
    for i = 1:2
        if channelState(i) == 1
            % Good -> Bad
            if rand < pGB
                channelState(i) = 0;
            end
        else
            % Bad -> Good
            if rand < pBG
                channelState(i) = 1;
            end
        end
    end

    % -------------------------------------------------------------
    % 2) Service step
    % -------------------------------------------------------------
    if action == 1 && q1 > 0
        if channelState(1) == 1
            pSucc = piGood(1);
        else
            pSucc = piBad(1);
        end

        if rand < pSucc
            q1 = q1 - 1;
            successFlag = 1;
        end

    elseif action == 2 && q2 > 0
        if channelState(2) == 1
            pSucc = piGood(2);
        else
            pSucc = piBad(2);
        end

        if rand < pSucc
            q2 = q2 - 1;
            successFlag = 1;
        end
    end

    % -------------------------------------------------------------
    % 3) Bernoulli arrivals
    % -------------------------------------------------------------
    q1 = q1 + (rand < lambda(1));
    q2 = q2 + (rand < lambda(2));

    % -------------------------------------------------------------
    % 4) Truncation
    % -------------------------------------------------------------
    q1 = min(q1, T);
    q2 = min(q2, T);

    Vnext = [q1; q2];
end

function action = baselinePolicy(V)
    % Baseline stabilizing policy beta_0:
    % longest-queue-first (LQF) with deterministic tie-breaker.

    if V(1) > V(2)
        action = 1;
    elseif V(2) > V(1)
        action = 2;
    else
        action = 1;
    end
end

function costVec = buildCostVector(T)
    % Cost C_f(V) = Q1 + Q2
    numStates = (T + 1)^2;
    costVec = zeros(numStates, 1);

    for q1 = 0:T
        for q2 = 0:T
            idx = sub2ind([T + 1, T + 1], q1 + 1, q2 + 1);
            costVec(idx) = q1 + q2;
        end
    end
end

function P_hat = estimateTransitionModel(transCounts)
    % Convert transition counts into smoothed transition probabilities.
    % P_hat(nextState, currentState, action)

    [S, ~, A] = size(transCounts);
    P_hat = zeros(S, S, A);

    for a = 1:A
        for s = 1:S
            c = transCounts(:, s, a);
            c = c + 1; % extra smoothing
            P_hat(:, s, a) = c / sum(c);
        end
    end
end

function [V, policy] = valueIterationMDP(P_hat, costVec, discountFactor, tol, maxIter)
    % Value iteration for the estimated truncated MDP.
    % Returns:
    %   V      - value function
    %   policy - optimal action per state (1 or 2)

    [S, ~, A] = size(P_hat);
    V = zeros(S, 1);
    policy = ones(S, 1);

    for iter = 1:maxIter
        Vnew = zeros(S, 1);

        for s = 1:S
            Q = zeros(A, 1);
            for a = 1:A
                Q(a) = costVec(s) + discountFactor * (P_hat(:, s, a)' * V);
            end

            [Vnew(s), policy(s)] = min(Q);
        end

        if max(abs(Vnew - V)) < tol
            V = Vnew;
            break;
        end

        V = Vnew;
    end
end