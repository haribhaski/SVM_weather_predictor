% Load the training data
clear all; clc;

train_data = readtable('Mumbai 1.csv');
X_train = table2array(train_data(:, 1:end-1)); % Input features
y_train = table2array(train_data(:, end)); % Target variable

% Preprocess the data (handle missing values, scale features, etc.)
% Identify numeric features
numeric_vars = varfun(@isnumeric, train_data, 'OutputFormat', 'uniform');
numeric_features = train_data.Properties.VariableNames(numeric_vars);

% Identify categorical features
categorical_features = train_data.Properties.VariableNames(~ismember(train_data.Properties.VariableNames, numeric_features));

% Convert non-ordinal categorical variables to ordinal
for i = 1:numel(categorical_features)
    train_data.(categorical_features{i}) = categorical(train_data.(categorical_features{i}));
end

% One-hot encode categorical features
dummyVarCoders = [];
for i = 1:numel(categorical_features)
    dummyVarCoders = [dummyVarCoders, dummyvar(train_data.(categorical_features{i}))];
end

% Concatenate numeric and one-hot encoded categorical features
X_numeric = table2array(train_data(:, numeric_features));
X_train = [X_numeric, dummyVarCoders];

% Convert the target variable to categorical if it's not already
if iscell(y_train)
    y_train = categorical(y_train);
end

% One-hot encode target variable
y_train_encoded = dummyvar((y_train));

% Scale numerical features
X_train_scaled = zscore(X_train);

% Train the multi-class SVM model using all the data
learning_rate = 0.0001;
num_iterations = 200;
regularization_param = 0.7;

svm_model = MultiClassMultiLabelSVM(learning_rate, num_iterations, regularization_param);
svm_model = svm_model.fit(X_train_scaled, y_train_encoded);

%%

fprintf('Select input method:\n');
fprintf('1. Provide your own data\n');
fprintf('2. Fetch live data from weather API\n');
input_choice = input('Enter your choice (1 or 2): ');

if(input_choice==1)
    % Scale the user input features
    fprintf('\nPlease provide input values for prediction:\n');
    for i = 1:numel(numeric_features)
        user_input(i) = input(sprintf('%s: ', numeric_features{i}));
    end

    % Scale the user input features
    user_input_scaled = (user_input - mean(X_numeric, 1)) ./ std(X_numeric, 0, 1);

elseif (input_choice==2)
        % API URL construction for current weather
        api_key = '3923d05fdf224c818da102707241405'; % Replace 'YOUR_API_KEY' with your actual API key
        city = 'Mumbai'; % Specify the city for which you want to fetch weather
        weather_url = sprintf('http://api.weatherapi.com/v1/current.json?key=%s&q=%s', api_key, city);
        
        % Make HTTP request for current weather
        weather_response = webread(weather_url);
        % Specify the API endpoint for forecast data
        forecast_url = sprintf('http://api.weatherapi.com/v1/forecast.json?key=%s&q=%s', api_key, city);
        % Make HTTP request for forecast data
        forecast_response = webread(forecast_url);
        % Access dew point from the forecast data

        dew_point_forecast = forecast_response.forecast.forecastday.hour.dewpoint_c; % Dew point in Celsius for each forecast entry
        dew_point = round(mean(dew_point_forecast, 'all'), 100); % Calculate mean across all elements of the matrix
        wg=forecast_response.forecast.forecastday.hour.gust_kph; % Dew point in Celsius for each forecast entry
        wind_gust = round(mean(wg, 'all'), 100); % Calculate mean across all elements of the matrix

        % Parse response for current weather
        temperature = weather_response.current.temp_c; % Temperature in Celsius
        humidity = weather_response.current.humidity; % Humidity in percentage
        wind_speed = round(weather_response.current.wind_kph,100); % Wind speed in kph
        sea_levelp= weather_response.current.pressure_mb; 
        prep= weather_response.current.precip_mm;
        weather_description = weather_response.current.condition.text; 
        
        % Display current weather information
        fprintf('\nCurrent weather in %s:\n', city);
        fprintf('Temperature: %.2f°C\n', temperature);
        fprintf('Dew Point: %d \n', dew_point);
        fprintf('Humidity: %d \n', humidity);
        fprintf('Precipitation: %d\n', prep);
        fprintf('Windgust: %d \n', wind_gust);
        fprintf('Wind Speed: %d \n', wind_speed);
        fprintf('Sea Level pressure: %d \n', sea_levelp);

        user_input(1) = temperature;
        user_input(2) = dew_point;
        user_input(3) = humidity;
        user_input(4) = prep; % Placeholder for precip, adjust as needed
        user_input(5) = wind_gust;
        user_input(6) = wind_speed;
        user_input(7) = sea_levelp;

else
    fprintf('Wrong Choice.. IT should be either 1 or 2');
end

% Scale the user input features
user_input_scaled = (user_input - mean(X_numeric, 1)) ./ std(X_numeric, 0, 1);

% Check the number of features in user input and the model
[num_user_features, num_user_samples] = size(user_input_scaled);
[num_model_features, num_classes] = size(svm_model.weights);

% If the number of features in user input is less than the model features,
% add zero-values for additional features
if num_user_features < num_model_features
    user_input_scaled = [user_input_scaled; zeros(num_model_features - num_user_features, num_user_samples)];
end

% If the number of columns in user_input_scaled is less than the number of columns in svm_model.weights
if size(user_input_scaled, 2) < size(svm_model.weights, 2)
    % Calculate the number of columns to add
    num_columns_to_add = size(svm_model.weights, 2) - size(user_input_scaled, 2);
    
    % Add columns of zeros to user_input_scaled
    user_input_scaled = [user_input_scaled, zeros(size(user_input_scaled, 1), num_columns_to_add)];
end

% Predict using the adjusted user input
user_prediction = svm_model.predict(user_input_scaled);

% Display the predicted label
[~, predicted_class] = max(user_prediction, [], 2);
predicted_class=mode(predicted_class);
predicted_category = categories(y_train);
predicted_label = predicted_category(predicted_class);

fprintf('\nPredicted label: %s\n', string(predicted_label));


