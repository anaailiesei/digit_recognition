function [X_train, y_train, X_test, y_test] = split_dataset(X, y, percent)
  % "Lipim y de X"
  append = [y, X];
  [m, n] = size(append);
  
  % Se permuta liniile
  perm = randperm(m);
  append_suffled = append(perm, :);
  
  % Numar linii pentru train
  m_train = round(percent * m);

  % Setul y si X de train
  append_train = append_suffled(1 : m_train, :);
  
  % Setul y si X de test
  % Setul y si X de test
  append_test = append_suffled(m_train + 1 : m, :);

  % Se "dezlipesc" y si X
  X_train = append_train(:, 2 : n);
  y_train = append_train(:, 1);
  X_test = append_test(:, 2 : n);
  y_test = append_test(:, 1);
endfunction
