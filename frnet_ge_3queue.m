% frlnet_policy_manager_ge_3q.m
% FRLNet Policy Manager simulation with:
%   - 3 queues
%   - time-varying Bernoulli arrival rates lambda_i(t)
%   - hidden Gilbert-Elliott wireless channels
%   - event-driven episodes
%   - exploration/exploitation policy manager
%
% This script is a more realistic second use-case for the FRLNet paper.
% The policy manager does NOT directly observe the channel states or the
% instantaneous arrival rates; it learns from the observed transitions.

clear; close all; clc;
rng(1); % Reproducibility

% -------------------------------------------------------------------------
% 1) Simulation parameters
% -------------------------------------------------------------------------

% Number of queues
Nq = 3;

% Arrival-rate base levels and amplitudes for lambda_i(t)
% The actual rates are generated as:
%   lambda_i(t) = base_i + amp_i * sin(2*pi*t/period + phase_i)
% and then clipped to [0, 0.95].
lambdaBase  = [0.28, 0.24, 0.18];
lambdaAmp   = [0.12, 0.10, 0.08];
lambdaPhase = [0.00, 1.20, 2.10];
lambdaPeriod = 250;

% Gilbert-Elliott wireless channel parameters
% channelState(i) = 1 -> Good state for link i
% channelState(i) = 0 -> Bad  state for link i
piGood = [0.90, 0.90, 0.90];
piBad  = [0.55, 0.55, 0.55];
pGB = 0.05;   % P(G -> B)
pBG = 0.15;   % P(B -> G)

% Queue truncation threshold (packets)
% Keep this modest because the state space grows as (T+1)^3.
T = 6;

% Practical low-backlog inner region approximation
gammaBound = 1;
T_in = max(T - gammaBound, 0);

% Policy manager hyperparameters
r = 0.5;                 % exploration-decay parameter
R = 10;                  % episode-transition threshold parameter
discountFactor = 0.98;    % discount factor for value iteration
valueTol = 1e-4;         % stopping tolerance
maxVIters = 500;         % max value-iteration iterations

% Simulation horizon and episode cap
totalSlotsTarget = 10000;
maxEpisodes = 500;

% -------------------------------------------------------------------------
% 2) State space and storage
% -------------------------------------------------------------------------

% Number of states in truncated 3-queue model
% q1,q2,q3 in {0,...,T} => (T+1)^3 states
numStates = (T + 1)^Nq;
numActions = Nq; % action a=1,2,3 => serve queue a

% Transition counts for empirical MDP estimation
% transCounts(nextState, currentState, action)
transCounts = ones(numStates, numStates, numActions);

% Traces
backlogTrace        = zeros(totalSlotsTarget, 1);
queueTrace          = zeros(totalSlotsTarget, Nq);
vmaxTrace           = zeros(totalSlotsTarget, 1);
episodeTrace        = zeros(totalSlotsTarget, 1);
phaseTrace          = zeros(totalSlotsTarget, 1); % 1 = explore, 2 = exploit
actionTrace         = zeros(totalSlotsTarget, 1);
channelStateTrace   = zeros(totalSlotsTarget, Nq);
successTrace        = zeros(totalSlotsTarget, 1);
lambdaTrace         = zeros(totalSlotsTarget, Nq);

% Initial backlog state V(t) = [Q1(t); Q2(t); Q3(t)]
V = zeros(Nq, 1);

% Initial channel state: start all links in Good state
channelState = ones(Nq, 1);

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

    % Cost is aggregate backlog: C_f(V) = Q1 + Q2 + Q3
    costVec = buildCostVector3(T);

    % Solve the estimated truncated MDP via value iteration
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

        % Time-varying arrival rates (hidden from the policy manager)
        lambdaNow = arrivalRatesAtTime( ...
            timeSlot, lambdaBase, lambdaAmp, lambdaPeriod, lambdaPhase);

        % Current backlog state
        q = V;

        % Map state to truncated MDP index
        sIdx = state2idx3(q, T);

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
                action = baselinePolicyLQF(V);
            end
            phase = 1;
        else
            % Model-based exploitation:
            % learned policy inside S_in, baseline policy outside S_in
            if isIn
                action = policyStar(sIdx);
            else
                action = baselinePolicyLQF(V);
            end
            phase = 2;
        end

        % ---------------------------------------------------------
        % One-step system evolution with Gilbert-Elliott channels
        % ---------------------------------------------------------
        [Vnext, channelState, successFlag] = simulateQueueStepGE3( ...
            V, action, lambdaNow, channelState, piGood, piBad, pGB, pBG, T);

        % Update transition counts for empirical model learning
        nextIdx = state2idx3(Vnext, T);
        transCounts(nextIdx, sIdx, action) = transCounts(nextIdx, sIdx, action) + 1;

        % Log traces
        backlogTrace(timeSlot)       = sum(V);
        queueTrace(timeSlot, :)      = V.';
        vmaxTrace(timeSlot)          = max(V);
        episodeTrace(timeSlot)       = episode;
        phaseTrace(timeSlot)         = phase;
        actionTrace(timeSlot)        = action;
        channelStateTrace(timeSlot,:) = channelState.';
        successTrace(timeSlot)       = successFlag;
        lambdaTrace(timeSlot,:)      = lambdaNow;

        % Count visits inside the low-backlog region
        if isIn
            episodeVisitsIn = episodeVisitsIn + 1;
        end

        % Advance to next state
        V = Vnext;

        % ---------------------------------------------------------
        % Episode termination condition
        % ---------------------------------------------------------
        if episodeVisitsIn >= R_theta
            episodeLengths(end + 1) = episodeSlotCounter; %#ok<SAGROW>
            episodeClosed = true;
            break;
        end
    end

    % If we hit the horizon before closing the episode,
    % still record the partial episode length.
    if ~episodeClosed && episodeSlotCounter > 0
        episodeLengths(end + 1) = episodeSlotCounter; %#ok<SAGROW>
    end

    % Move to next episode
    episode = episode + 1;
end

% Trim traces to actual simulated length
validIdx = 1:timeSlot;
backlogTrace        = backlogTrace(validIdx);
queueTrace          = queueTrace(validIdx, :);
vmaxTrace           = vmaxTrace(validIdx);
episodeTrace        = episodeTrace(validIdx);
phaseTrace          = phaseTrace(validIdx);
actionTrace         = actionTrace(validIdx);
channelStateTrace   = channelStateTrace(validIdx, :);
successTrace        = successTrace(validIdx);
lambdaTrace         = lambdaTrace(validIdx, :);

% -------------------------------------------------------------------------
% 4) Plots
% -------------------------------------------------------------------------

% Aggregate backlog
figure;
plot(validIdx, backlogTrace, 'LineWidth', 1.8);
xlabel('Time slot', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('Aggregate backlog \Sigma_i Q_i', 'FontSize', 14, 'FontWeight', 'bold');
grid on;
ax = gca;
ax.FontSize = 12;
ax.FontWeight = 'bold';
set(gca, 'LooseInset', max(get(gca, 'TightInset'), 0.02));
exportgraphics(gcf, 'frlnet_3q_backlog_trace.png', 'Resolution', 600);

% Per-queue backlog
figure;
plot(validIdx, queueTrace(:,1), 'LineWidth', 1.5); hold on;
plot(validIdx, queueTrace(:,2), 'LineWidth', 1.5);
plot(validIdx, queueTrace(:,3), 'LineWidth', 1.5);
xlabel('Time slot', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('Queue backlog', 'FontSize', 14, 'FontWeight', 'bold');
legend({'Q_1(t)', 'Q_2(t)', 'Q_3(t)'}, 'Location', 'best', 'FontSize', 12);
grid on;
ax = gca;
ax.FontSize = 12;
ax.FontWeight = 'bold';
set(gca, 'LooseInset', max(get(gca, 'TightInset'), 0.02));
exportgraphics(gcf, 'frlnet_3q_queue_backlog.png', 'Resolution', 600);

% Channel-state plot
figure;
stairs(validIdx, channelStateTrace(:,1), 'LineWidth', 1.2); hold on;
stairs(validIdx, channelStateTrace(:,2), 'LineWidth', 1.2);
stairs(validIdx, channelStateTrace(:,3), 'LineWidth', 1.2);
ylim([-0.2 1.2]);
xlabel('Time slot', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('Channel state', 'FontSize', 14, 'FontWeight', 'bold');
legend({'Link 1', 'Link 2', 'Link 3'}, 'Location', 'best', 'FontSize', 12);
grid on;
ax = gca;
ax.FontSize = 12;
ax.FontWeight = 'bold';
set(gca, 'LooseInset', max(get(gca, 'TightInset'), 0.02));
exportgraphics(gcf, 'frlnet_3q_channels.png', 'Resolution', 600);

% Time-varying arrival rates
figure;
plot(validIdx, lambdaTrace(:,1), 'LineWidth', 1.4); hold on;
plot(validIdx, lambdaTrace(:,2), 'LineWidth', 1.4);
plot(validIdx, lambdaTrace(:,3), 'LineWidth', 1.4);
xlabel('Time slot', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('Arrival rate \lambda_i(t)', 'FontSize', 14, 'FontWeight', 'bold');
legend({'\lambda_1(t)', '\lambda_2(t)', '\lambda_3(t)'}, 'Location', 'best', 'FontSize', 12);
grid on;
ax = gca;
ax.FontSize = 12;
ax.FontWeight = 'bold';
set(gca, 'LooseInset', max(get(gca, 'TightInset'), 0.02));
exportgraphics(gcf, 'frlnet_3q_arrival_rates.png', 'Resolution', 600);

% Episode lengths
figure;
plot(episodeLengths, '-o', 'LineWidth', 1.8);
xlabel('Episode index', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('Episode length (slots)', 'FontSize', 14, 'FontWeight', 'bold');
grid on;
ax = gca;
ax.FontSize = 12;
ax.FontWeight = 'bold';
set(gca, 'LooseInset', max(get(gca, 'TightInset'), 0.02));
exportgraphics(gcf, 'frlnet_3q_episode_lengths.png', 'Resolution', 600);

% -------------------------------------------------------------------------
% 5) Console summary
% -------------------------------------------------------------------------
fprintf('Simulation finished.\n');
fprintf('  Total slots simulated : %d\n', timeSlot);
fprintf('  Episodes completed    : %d\n', episode - 1);
fprintf('  Mean aggregate backlog: %.3f packets\n', mean(backlogTrace));
fprintf('  Final state           : [%d, %d, %d]\n', V(1), V(2), V(3));

% =========================================================================
% Local helper functions
% =========================================================================

function lambdaNow = arrivalRatesAtTime(t, lambdaBase, lambdaAmp, period, phase)
    % Time-varying Bernoulli arrival rates lambda_i(t).
    % Values are clipped to [0.02, 0.95] to stay valid probabilities.

    Nq = numel(lambdaBase);
    lambdaNow = zeros(1, Nq);

    for i = 1:Nq
        raw = lambdaBase(i) + lambdaAmp(i) * sin(2*pi*t/period + phase(i));
        lambdaNow(i) = min(max(raw, 0.02), 0.95);
    end
end

function idx = state2idx3(V, T)
    % Map 3-queue state V = [q1;q2;q3] to a unique index.
    q1 = min(max(round(V(1)), 0), T);
    q2 = min(max(round(V(2)), 0), T);
    q3 = min(max(round(V(3)), 0), T);
    idx = sub2ind([T + 1, T + 1, T + 1], q1 + 1, q2 + 1, q3 + 1);
end

function [Vnext, channelState, successFlag] = simulateQueueStepGE3( ...
    V, action, lambdaNow, channelState, piGood, piBad, pGB, pBG, T)
    % One-step queue evolution under Gilbert-Elliott wireless channels.
    %
    % The policy manager does NOT directly observe channelState.
    % It only sees the resulting transition behavior.

    Nq = 3;
    successFlag = 0;
    q = V(:);

    % -------------------------------------------------------------
    % 1) Channel evolution (hidden from the scheduler)
    % -------------------------------------------------------------
    for i = 1:Nq
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
    if action >= 1 && action <= Nq && q(action) > 0
        if channelState(action) == 1
            pSucc = piGood(action);
        else
            pSucc = piBad(action);
        end

        if rand < pSucc
            q(action) = q(action) - 1;
            successFlag = 1;
        end
    end

    % -------------------------------------------------------------
    % 3) Bernoulli arrivals
    % -------------------------------------------------------------
    for i = 1:Nq
        q(i) = q(i) + (rand < lambdaNow(i));
    end

    % -------------------------------------------------------------
    % 4) Truncation
    % -------------------------------------------------------------
    q = min(q, T);

    Vnext = q;
end

function action = baselinePolicyLQF(V)
    % Baseline stabilizing policy beta_0:
    % longest-queue-first with deterministic tie-breaking.

    [~, action] = max(V); % returns first max if tie
end

function costVec = buildCostVector3(T)
    % Cost C_f(V) = Q1 + Q2 + Q3

    numStates = (T + 1)^3;
    costVec = zeros(numStates, 1);

    for q1 = 0:T
        for q2 = 0:T
            for q3 = 0:T
                idx = sub2ind([T + 1, T + 1, T + 1], q1 + 1, q2 + 1, q3 + 1);
                costVec(idx) = q1 + q2 + q3;
            end
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
            c = c + 1; % smoothing
            P_hat(:, s, a) = c / sum(c);
        end
    end
end

function [V, policy] = valueIterationMDP(P_hat, costVec, discountFactor, tol, maxIter)
    % Value iteration for the estimated truncated MDP.

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