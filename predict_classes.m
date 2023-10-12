function [classes] = predict_classes(X, weights, ...
                  input_layer_size, hidden_layer_size, ...
                  output_layer_size)
  format long

  % Creeaza matricele theta 1 si theta 2
  numel_theta1 = (input_layer_size + 1) * hidden_layer_size;
  weights_theta1 = weights(1 : numel_theta1);
  theta1 = reshape(weights_theta1, hidden_layer_size, input_layer_size + 1);
  weights_theta2 = weights(numel_theta1 + 1 : end);
  theta2 = reshape(weights_theta2, output_layer_size, hidden_layer_size + 1);

  % Forward propagation
  %Prin input layer
  m = rows(X);
  a_1 = [ones(m, 1), X];
  a_1 = a_1';

  % Prin hidden layer
  z_2 = theta1 * a_1;
  a_2 = sigmoid(z_2);
  a_2 = [ones(1, m); a_2];

  % Prin output layer
  z_3 = theta2 * a_2;
  a_3 = sigmoid(z_3);
  a_3 = a_3';

  % Se cauta indexul probabilitatii maxime din fiecare coloana
  [~, classes] = max(a_3, [], 2);
endfunction
