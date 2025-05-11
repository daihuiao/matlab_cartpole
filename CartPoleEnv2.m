classdef CartPoleEnv2 < rl.env.MATLABEnvironment % 或 classdef CartPoleEnv1 < rl.env.MATLABEnvironment
    properties
        DynParams = struct(...
            'mc', 1.0, 'mp', 0.1, 'l', 0.5, 'g', 9.81, 'b', 0.1, 'd', 0.05);
        Ts = 0.05;
        State = zeros(4,1); % 内部状态: [x, theta, x_dot, theta_dot]
        PositionThreshold = 2.4;
        MaxStepsPerEpisode = 200;
        EnableVisualizer = false;
        FigureHandle;
        steps_my;
        EpisodeTrajectory;
    end

    properties(Access = protected)
        IsDone = false;
    end

    methods
        function this = CartPoleEnv2() % 或 CartPoleEnv1
            % 为 ObservationInfo 定义常量或直接使用数值
            obsDim = 5; 
            defaultPosThreshold = 2.4; % 使用局部变量或直接数值
            obsLow  = [-defaultPosThreshold; -inf; -1; -1; -inf];
            obsHigh = [ defaultPosThreshold;  inf;  1;  1;  inf];
            ObservationInfo = rlNumericSpec([obsDim 1], 'LowerLimit', obsLow, 'UpperLimit', obsHigh);
            ObservationInfo.Name = 'Cart-Pole Swing-Up State';
            ObservationInfo.Description = 'x, x_dot, cos(theta), sin(theta), theta_dot';

            actLow = -100;
            actHigh = 100;
            ActionInfo = rlNumericSpec([1 1], 'LowerLimit', actLow, 'UpperLimit', actHigh);
            ActionInfo.Name = 'Cart Force';

            this = this@rl.env.MATLABEnvironment(ObservationInfo, ActionInfo); % 超类构造函数调用

            % 后续初始化
            if ~isfield(this.DynParams, 'd')
                this.DynParams.d = 0.0;
            end
            % 初始状态将在 reset 方法中设置
        end

        function [Observation, Reward, IsDone, LoggedSignals] = step(this, Action)
            if any(isnan(Action(:))) || any(isinf(Action(:)))
                fprintf('NaN or Inf detected in Action!\n');
                % keyboard; % Pause execution for debugging
            end
            % Action = 0.0;
            % Action = Action * 10;
            LoggedSignals = [];
            X = this.State;
            % 确保 Action 是标量；getActionInfo(this) 也可以在构造函数中存为属性以避免重复调用
            u = max(this.ActionInfo.LowerLimit, min(this.ActionInfo.UpperLimit, Action(1)));

            time_for_dynamics = 0;
            xdot = cartpole_dynamics_local(time_for_dynamics, X, u, this.DynParams);
            
            X_next = X + xdot * this.Ts;
            this.State = X_next;

            x_cart    = this.State(1);
            theta     = this.State(2); 
            x_dot     = this.State(3);
            theta_dot = this.State(4);

            Observation = [x_cart; x_dot; cos(theta); sin(theta); theta_dot];

            this.EpisodeTrajectory(:, this.steps_my + 1) = this.State;
            this.steps_my = this.steps_my + 1;

            reward_height = -0.1*cos(theta); 
            % penalty_cart_pos = 0.5 * (x_cart / this.PositionThreshold)^2;
            % penalty_theta_dot = 0.05 * theta_dot^2;
            % penalty_action = 0.001 * u^2;
            % Reward = reward_height - penalty_cart_pos - penalty_theta_dot - penalty_action;
            Reward = reward_height;

            IsDone = false;
            if abs(x_cart) > this.PositionThreshold
                IsDone = true;
                Reward = Reward - 50; % 对出界给予较大惩罚 animate_cartpole_motion_new(this.EpisodeTrajectory,this.DynParams)
            end
            if abs(x_cart) > 0.9*this.PositionThreshold
                % IsDone = true;
                Reward = Reward - 10; % 对出界给予较大惩罚 animate_cartpole_motion_new(this.EpisodeTrajectory,this.DynParams)
            end
            this.IsDone = IsDone;

            if this.EnableVisualizer
                this.render();
            end

            if any(isnan(Observation(:))) || any(isinf(Observation(:)))
                fprintf('NaN or Inf detected in nextObservation!\n');
                keyboard; % Pause execution for debugging
            end
            if isnan(Reward) || isinf(Reward)
                fprintf('NaN or Inf detected in reward!\n');
                keyboard;
            end

        end

        function InitialObservation = reset(this)
            this.steps_my = 0;
            this.EpisodeTrajectory = NaN(4, this.MaxStepsPerEpisode + 1); 
            initial_theta = 0; % 杆子从下方开始
            this.State = [0; initial_theta; 0; 0];

            x_cart    = this.State(1);
            theta     = this.State(2);
            x_dot     = this.State(3);
            theta_dot = this.State(4);
            InitialObservation = [x_cart; x_dot; cos(theta); sin(theta); theta_dot];
            this.EpisodeTrajectory(:, 1) = this.State;
            
            this.IsDone = false;
            
            if this.EnableVisualizer && ~isempty(this.FigureHandle) && isvalid(this.FigureHandle)
                try close(this.FigureHandle); catch; end
            end
            this.FigureHandle = [];
        end

        function render(this, varargin)
            if isempty(this.FigureHandle) || ~isvalid(this.FigureHandle)
                this.FigureHandle = figure('Name', 'Cart-Pole Swing-Up', 'Visible', 'on');
                ax = gca(this.FigureHandle);
                title(ax, 'Cart-Pole Swing-Up'); xlabel(ax, 'Cart Pos (m)'); ylabel(ax, 'Vertical (m)');
                grid(ax, 'on'); axis(ax, 'equal');
                pole_len = this.DynParams.l * 2;
                xlim(ax, [-this.PositionThreshold*1.5, this.PositionThreshold*1.5]);
                ylim(ax, [-pole_len*1.2, pole_len*1.2]);
            end
            
            try figure(this.FigureHandle); catch, return; end
            
            currentTime = 0; 
            
            if exist('draw_cartpole', 'file') ~= 2 % 检查 draw_cartpole 是否存在
                clf(this.FigureHandle); hold(gca, 'on');
                cart_x = this.State(1); pole_theta = this.State(2); L_pole = this.DynParams.l * 2;
                cart_w = 0.4; cart_h = 0.2; pivot_y_vis = 0;
                rectangle(gca, 'Position', [cart_x - cart_w/2, pivot_y_vis - cart_h/2, cart_w, cart_h], 'FaceColor', 'b');
                pole_x_end = cart_x + L_pole * sin(pole_theta);
                pole_y_end = pivot_y_vis + L_pole * cos(pole_theta); % theta=0 向上, theta=pi 向下
                line(gca, [cart_x, pole_x_end], [pivot_y_vis, pole_y_end], 'Color', 'r', 'LineWidth', 3);
                title(gca, 'Cart-Pole Swing-Up (Basic)'); xlabel(gca, 'Cart Pos (m)'); ylabel(gca, 'Vertical (m)');
                grid(gca, 'on'); axis(gca, 'equal');
                xlim(gca, [-this.PositionThreshold*1.5, this.PositionThreshold*1.5]);
                ylim(gca, [-L_pole*1.2, L_pole*1.2]);
                hold(gca, 'off');
            else % 如果 draw_cartpole 存在，则调用它
                 draw_cartpole(currentTime, this.State, this.DynParams);
            end
            % --- 基础绘图代码结束 ---

            drawnow limitrate;
        end
    end
end

% function xdot = cartpole_dynamics_local(t,x,u,param)
%     mc = param.mc; mp = param.mp; l = param.l; g = param.g; b = param.b; d = param.d;
%     x1=x(1); x2=x(2); x3=x(3); x4=x(4);
%     s = sin(x2); c = cos(x2);
%     xdot = zeros(4,1);
% 
%     den_val = mc + mp*s^2;
%     if abs(den_val) < 1e-8, den_val = 1e-8 * sign(den_val); end % 避免除以过小值
% 
%     xdot(3) = (u - b*x3 + mp*s*(l*x4^2 + g*c)) / den_val;
% 
%     common_den_dynamics = l * den_val;
%     if abs(common_den_dynamics) < 1e-8, common_den_dynamics = 1e-8 * sign(common_den_dynamics); end
% 
%     numerator_theta_ddot_orig = -u*c + b*x3*c - d*(mc+mp)*x4/(mp*l) - mp*l*(x4^2)*c*s - (mc+mp)*g*s/(mp*l);
%     xdot(4) = numerator_theta_ddot_orig / common_den_dynamics;
% 
%     xdot(1) = x3;
%     xdot(2) = x4;
% end

function xdot=cartpole_dynamics_local(t,x,u,param)

mc = param.mc;
mp = param.mp;
l = param.l;
g=param.g;
b=param.b;
d=param.d;

x1=x(1);
x2=x(2);
x3=x(3);
x4=x(4);

s = sin(x(2)); c = cos(x(2));

xdot(1,:)=x(3);
xdot(2,:)=x(4);
xdot(3,:) = [u-b*x3+d*x4*c/l+ mp*s*(l*x4^2 + g*c)]/[mc+mp*s^2];
xdot(4,:) =[-u*c+b*x3*c-d*(mc+mp)*x4/(mp*l) - mp*l*x4^2*c*s - (mc+mp)*g*s]/[l*(mc+mp*s^2)];
% xdot(4,:) =[-u*c+b*x3*c-d*(mc+mp)*x4/(mp*l) - mp*l*x4^2*c*s - (mc+mp)*g*s/(mp*l)]/[l*(mc+mp*s^2)];

end



function draw_cartpole__(t,x,param)
      l=param.l; % 从参数中获取摆杆长度
      persistent hFig base a1 raarm wb lwheel rwheel av theta_wheel_angles aw wheelr; % 将绘图参数设为持久化

      % ========================== 关键修改处 ==========================
      % 检查 hFig 是否为空，或者是否不再是一个有效的图形窗口句柄
      if (isempty(hFig) || ~isgraphics(hFig, 'figure'))
          % 如果 hFig 为空 (通常是第一次调用)
          % 或者 hFig 不是一个有效的图形窗口句柄 (例如，之前的窗口被关闭了)
          % 则重新创建一个图形窗口及所有相关的持久化绘图元素

          hFig = figure(25); % 您可以使用 figure() 来总是创建新窗口，或指定一个编号
          set(hFig,'DoubleBuffer', 'on');
          
          % --- 所有只需要在图形创建时计算一次的参数和形状定义 ---
          a1 = l+0.25;
          av = pi*[0:.05:1];
          theta_wheel_angles = pi*[0:0.05:2]; % 用于轮子定义，避免与状态x(2)的theta混淆
          wb = .3; hb=.15;
          aw = .01;
          wheelr = 0.05;
          lwheel = [-wb/2 + wheelr*cos(theta_wheel_angles); -hb-wheelr + wheelr*sin(theta_wheel_angles)]';
          % rwheel 变量已声明为 persistent 但未在此处单独定义，左右轮似乎共用 lwheel 定义并平移
          base = [wb*[1 -1 -1 1]; hb*[1 1 -1 -1]]';
          arm = [aw*cos(av-pi/2) -a1+aw*cos(av+pi/2); ...
                 aw*sin(av-pi/2) aw*sin(av+pi/2)]';
          raarm = [(arm(:,1).^2+arm(:,2).^2).^.5, atan2(arm(:,2),arm(:,1))];
          % --- 持久化元素定义结束 ---
      end
      % ========================== 修改结束 ===========================
      
      % 确保图形窗口句柄有效后才继续 (额外的安全检查)
      if ~isgraphics(hFig, 'figure')
          warning('无法创建或激活有效的图形句柄。动画可能无法显示。');
          return; 
      end
      
      figure(hFig); % 激活图形窗口
      cla;          % 清除当前坐标区
      hold on;      % 保持当前图形，以便添加新的绘图对象
      view(0,90);   % 设置2D视角
      
      % 绘制小车底座
      patch(x(1)+base(:,1), base(:,2),0*base(:,1),'b','FaceColor',[.3 .6 .4]);
      % 绘制轮子
      patch(x(1)+lwheel(:,1), lwheel(:,2), 0*lwheel(:,1),'k');
      patch(x(1)+wb+lwheel(:,1), lwheel(:,2), 0*lwheel(:,1),'k'); % 右轮通过平移左轮形状得到
      % 绘制摆杆 (arm)
      patch(x(1)+raarm(:,1).*sin(raarm(:,2)+x(2)-pi),-raarm(:,1).*cos(raarm(:,2)+x(2)-pi), 1+0*raarm(:,1),'r','FaceColor',[.9 .1 0]);
      % 绘制摆杆末端标记
      plot3(x(1)+l*sin(x(2)), -l*cos(x(2)),1, 'ko',...
        'MarkerSize',10,'MarkerFaceColor','b');
      % 绘制小车上的一个点 (可能是摆杆的枢轴点或其他参考)
      plot3(x(1),0,1.5,'k.');
      
      title(['t = ', num2str(t,'%.2f') ' sec']); % 设置标题显示时间
      set(gca,'XTick',[],'YTick',[]); % 移除坐标轴刻度
      
      axis image; % 使x,y轴单位长度相同，避免变形
      axis([-2.5 2.5 -2.5*l 2.5*l]); % 设置坐标轴范围
      
      drawnow; % 强制刷新图形窗口，显示当前帧
end


function animate_cartpole_motion_new(x_trajectory, param, dt)
% animate_cartpole_motion:
% 播放预先计算好的推车摆（cart-pole）的动作轨迹动画。
% animate_cartpole_motion(this.EpisodeTrajectory,this.DynParams) % (这是您注释中的示例用法)
%
% 输入参数:
%   x_trajectory: 状态轨迹矩阵，大小为 [状态数量, 步数N]。
%                 每一列 x_trajectory(:,k) 是第 k 步的状态
%                 [小车位置; 摆杆角度; 小车速度; 摆杆角速度]。
%   param:        draw_cartpole 函数所需的参数结构体 (必须包含 param.l - 摆杆长度)。
%   dt:           动画播放时每一步的暂停时间 (秒)。
%                 如果未提供，则默认为 0.05 秒。

    % 检查 dt 是否提供，若未提供则设置默认值
    if nargin < 3 || isempty(dt)
        dt = 0.05; % 动画默认的时间步长
    end

    % 初始检查轨迹是否完全为空
    if isempty(x_trajectory)
        disp('输入轨迹 x_trajectory 为空，无法播放动画。');
        return;
    end

    % ==> 新增：处理 NaN 值 <==
    % 查找第一个至少包含一个 NaN 值的列的索引
    nan_column_flags = any(isnan(x_trajectory), 1); % 得到一个逻辑行向量，标记哪些列包含NaN
    first_nan_column_index = find(nan_column_flags, 1, 'first');

    if ~isempty(first_nan_column_index)
        % 如果找到了包含 NaN 的列，则只取该列之前的部分
        if first_nan_column_index == 1
            % 如果第一列就包含 NaN，说明整个轨迹无效或为空
            disp('轨迹从第一列开始就包含NaN，或处理后为空，无法播放动画。');
            return;
        end
        % 保留从第一列到第一个NaN列之前一列的数据
        x_trajectory = x_trajectory(:, 1:first_nan_column_index-1);
    end
    % ==> NaN 处理结束 <==

    % 获取处理后轨迹的步数
    [num_states, N_steps] = size(x_trajectory);

    % 再次检查处理后的轨迹是否为空
    if N_steps == 0
        disp('有效轨迹为空 (移除NaN后轨迹为空)，无法播放动画。');
        return;
    end

    % 可选: 检查状态维度是否符合预期 (draw_cartpole 通常使用4个状态)
    if num_states < 2 % 至少需要位置和角度来进行基础绘图
        error('状态轨迹 x_trajectory 的行数不足以进行绘图 (至少需要2行表示位置和角度)。');
    elseif num_states < 4
        warning('状态轨迹 x_trajectory 的行数少于4行，draw_cartpole 可能无法使用所有期望的状态。');
    end
    
    % 创建时间向量用于显示
    tVec = (0:N_steps-1) * dt;

    % 循环播放轨迹中的每一步
    disp('开始播放动画...'); % 添加一个开始提示
    for k = 1:N_steps
        current_state = x_trajectory(:, k);
        current_time = tVec(k);
        
        % 调用您提供的 draw_cartpole 函数
        % 注意: 您在您提供的 animate_cartpole_motion 代码中写的是 draw_cartpole_
        % 请确保这里的函数名与您的实际绘图函数名一致。
        % 我将假设您的函数名是 draw_cartpole (没有下划线)
        if exist('draw_cartpole__', 'file') == 2
            draw_cartpole__(current_time, current_state, param); % 如果您的函数确实是 draw_cartpole_
        elseif exist('draw_cartpole', 'file') == 2
            draw_cartpole(current_time, current_state, param);  % 或者标准的 draw_cartpole
        else
            warning('绘图函数 draw_cartpole (或 draw_cartpole_) 未找到。动画将无法显示。');
            return; % 如果找不到绘图函数，则退出
        end
        
        % 暂停以控制动画速度
        pause(dt);
    end
    disp('动画播放完毕。');
end