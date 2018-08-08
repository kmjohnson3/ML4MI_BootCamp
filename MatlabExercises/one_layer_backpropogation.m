clear
clc

Nexamples = 1000;
x = linspace(-5,5,Nexamples)';
fx = sin(x);
f2x= sin(2*x);
fx_name ='Sin(x)'
N = 200; % number of activations
batch_size = 512;

% Initialize weights (this is not actually the best initializer)
b = randn(N,1);
win = randn(N,1);
wout = randn(N,1);
loss = [];

figure('Position',[100 100 1000 500])
for iter = 1:10000
    
    idx = randi(Nexamples,[batch_size,1]);
    input_mini_batch = x(idx);
    
    % forward pass
    input = input_mini_batch;
    for i = 1:N
        hidden_layer{i} = max(0,win(i)*input + b(i));
    end
    
    output = zeros(batch_size,1);
    for i = 1:N
        output = output + hidden_layer{i}*wout(i);
    end
    
    % error estimate, this is dE/dO (O for output)
    error = 2*(output - fx(idx));
    
    loss(iter) = sum(error.^2);
    
    % Plots for visualization
    if mod(iter,10)==0
        subplot(121)
        
        % Do all the training data
        inputFull = x;
        for i = 1:N
            hidden_layerFull{i} = max(0,win(i)*inputFull + b(i));
        end
        outputFull = zeros(Nexamples,1);
        for i = 1:N
            outputFull = outputFull + hidden_layerFull{i}*wout(i);
        end
        
        plot(x,fx,inputFull,outputFull,'LineWidth',4);
        xlabel('X');
        ylabel('f(X)')
        legend(fx_name,'Prediction')
        legend('boxoff');
        set(gca,'LineWidth',5,'FontSize',16)
        subplot(122)
        semilogy(loss,'LineWidth',4)
        xlabel('Iteration');
        ylabel('Batch Loss')
        set(gca,'LineWidth',5,'FontSize',16)
        drawnow
    end
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % back propogation ( Relu has derivative 0 or 1
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Last layer is just a O=w*I ( I is input from previous layer)
    % using the chain rule:
    %  dE/dWout =   ( dE/dO ) * (      dO/dW     )
    for i = 1:N
        grad_wout{i} =  error   .*   hidden_layer{i};
    end
    
    % Bias - Goes across Relu
    % dE/dB = ( dE/dO )( dO/dB )
    % dO/dB = wout*1 or zero if the nueron is not activated
    for i = 1:N
        %          ( dE/DO) (            dO/dB              )
        grad_b{i} = error .* wout(i).*( hidden_layer{i} > 0);
    end
    
    %  Weight - Goes across Relu
    % dE/dWin = ( dE/dO )( dO/dWin )
    for i = 1:N
        %            ( dE/dO )
        grad_win{i} = error.*wout(i).*input.*( hidden_layer{i} > 0);
    end
    
    % Now we update based on a learning rate
    lr = 1e-3; % Lower rate
    for i = 1:N
        win(i) = win(i) - lr*mean( grad_win{i} );
        wout(i) =wout(i) -  lr*mean( grad_wout{i} );
        b(i) = b(i) - lr*mean( grad_b{i} );
    end
end

% Compute outside the function bounds
% Do all the training data
inputFull = 2*x;
for i = 1:N
    hidden_layerFull{i} = max(0,win(i)*inputFull + b(i));
end
outputFull = zeros(Nexamples,1);
for i = 1:N
    outputFull = outputFull + hidden_layerFull{i}*wout(i);
end

figure
plot(2*x,f2x,inputFull,outputFull,'LineWidth',4);
xlabel('X');
ylabel('f(X)')
legend(fx_name,'Prediction')
legend('boxoff');
set(gca,'LineWidth',5,'FontSize',16)


