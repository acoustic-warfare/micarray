function index = weight_index(matrix_array,frequency,c)

uni_distance = matrix_array.uni_distance;

lambda = c/frequency;

lambda_rel = uni_distance/lambda;

if lambda_rel > 0.1581
    % mode = 1
    index = 1;
    return
elseif 0.156>= lambda_rel && lambda_rel > 0.0986
    index =3;
elseif 0.0986 >= lambda_rel && lambda_rel > 0.085
    index = 5;
elseif 0.085 >= lambda_rel && lambda_rel > 0.07
    index = 6;
else
    index = 7;
end



end

