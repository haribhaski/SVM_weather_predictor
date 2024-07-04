classdef MultiClassMultiLabelSVM
    properties
        learning_rate
        num_iterations
        regularization_param % Regularization parameter (lambda)
        weights
        biases
        num_samples
        num_features
        num_classes
         % Threshold for binary classification
    end
    
    methods
        function obj = MultiClassMultiLabelSVM(learning_rate, num_iterations, regularization_param)

            obj.learning_rate = learning_rate;
            obj.num_iterations = num_iterations;
            obj.regularization_param = regularization_param;
        end
        
        
        % Training function
        function obj = fit(obj, X, Y)
            % Initialize model parameters
            obj.num_samples = size(X, 1);
            obj.num_features = size(X, 2);
            obj.num_classes = size(Y, 2);
            obj.weights = randn(obj.num_classes, size(X, 2)) * sqrt(2 / size(X, 2)); % Xavier initialization
            obj.biases = randn(1, obj.num_classes) * 0.01; % Small random biases
            
            % Mini-batch gradient descent
            batch_size = 50;
            num_batches = ceil(obj.num_samples / batch_size);
            
            for iter = 1:obj.num_iterations
                total_loss = 0;
                
                % Adjust learning rate
                current_learning_rate = obj.learning_rate / (1 + 0.001 * iter);
                
                % Iterate over batches
                for batch = 1:num_batches
                    start_idx = (batch - 1) * batch_size + 1;
                    end_idx = min(batch * batch_size, obj.num_samples);
                    X_batch = X(start_idx:end_idx, :);
                    Y_batch = Y(start_idx:end_idx, :);
                    
                    % Compute loss and update weights
                    for i = 1:size(X_batch, 1)
                        x_i = X_batch(i, :);
                        y_i = Y_batch(i, :);
                        
                        for class_idx = 1:obj.num_classes
                            target = y_i(class_idx);
                            prediction = dot(obj.weights(class_idx, :), x_i) + obj.biases(class_idx);
                            loss = max(0, 1 - target * prediction); % Hinge loss
                            total_loss = total_loss + loss;
                            
                            if loss > 0
                                % Compute gradient and update weights
                                gradient = -target * x_i;
                                gradient = min(max(gradient, -5), 5); % Gradient clipping
                                obj.weights(class_idx, :) = obj.weights(class_idx, :) - current_learning_rate * (gradient + obj.regularization_param * obj.weights(class_idx, :));
                                obj.biases(class_idx) = obj.biases(class_idx) - current_learning_rate * (-target);
                            end
                        end
                    end
                end
                
                % Print average loss every few iterations
                avg_loss = total_loss / (obj.num_samples * obj.num_classes);
                if mod(iter, 10) == 0
                    fprintf('Iteration %d: Avg loss: %.4f\n', iter, avg_loss);
                end
            end
        end
        
        % Prediction function
        function predicted_labels = predict(obj, X_test, class_names)
            num_test_samples = size(X_test, 1);
            num_classes = obj.num_classes;
            predicted_labels = zeros(num_test_samples, num_classes);
        
            for i = 1:num_test_samples
                x_test = X_test(i, :);
                scores = zeros(1, num_classes);
                for class_idx = 1:num_classes
                    scores(class_idx) = dot(obj.weights(class_idx, :), x_test) + obj.biases(class_idx);
                end
                predicted_labels(i, :) = scores;
            end
        end
        
    end
end
