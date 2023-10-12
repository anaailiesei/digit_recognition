function [J, grad] = cost_function(params, X, y, lambda, ...
                   input_layer_size, hidden_layer_size, ...
                   output_layer_size)
  format long
  
  % Creeaza matricele theta 1 si theta 2
  numel_theta1 = (input_layer_size + 1) * hidden_layer_size;
  params_theta1 = params(1 : numel_theta1);
  theta1 = reshape(params_theta1, hidden_layer_size, input_layer_size + 1);
  params_theta2 = params(numel_theta1 + 1 : end);
  theta2 = reshape(params_theta2, output_layer_size, hidden_layer_size + 1);
  
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

  % Output
  a_3 = sigmoid(z_3);

  % Backpropagation
  n = length(y);
  
  % Se expandeaza y
  y_exp = eye(10)(y, :);

  % Se calculeaza delta 2
  del_3 = a_3 - y_exp';
  delta_2 = del_3 * a_2';
  
  % Se calculeaza sigma derivat de z_2
  n = rows(a_2);
  a_2 = a_2(2 : n, :);
  sigm_deriv = a_2 .* (ones(size(a_2)) - a_2);

  % Se elimina primul rand din primul operand al lui del 2
  first_op = theta2' * del_3;
  n = rows(first_op);
  first_op = first_op(2 : n, :);
  del_2 = first_op .* sigm_deriv;

  % Se calculeaza delta 1
  delta_1 = del_2 * a_1';

  % Se determina gradientii
  grad_1 = delta_1 / m;
  grad_1(:, 2 : end) += lambda / m * theta1(:, 2 : end);
  
  grad_2 = delta_2 / m;
  grad_2(:, 2 : end) += lambda / m * theta2(:, 2 : end);

  % Functia de cost si gardientul final
  % Gradientul
  vec1 = reshape(grad_1, [], 1);
  vec2 = reshape(grad_2, [], 1);
  grad = vertcat(vec1, vec2);

  % Pentru cost
  % Se calculeaza prima suma
  summation_matrix_1 = -y_exp' .* log(a_3) - ~y_exp' .* log(1 - a_3);
  sum_1 = sum(summation_matrix_1(:));
  
  % Se calculeaza a doua suma
  summation_matrix_2 = theta1(:, 2 : end) .^ 2;
  sum_2 = sum(summation_matrix_2(:));
  
  % Se calculeaza a treia suma
  summation_matrix_3 = theta2(:, 2 : end) .^ 2;
  sum_3 = sum(summation_matrix_3(:));
  
  % J final
  J = sum_1 / m + lambda * (sum_2 + sum_3) / 2 / m;
endfunction
