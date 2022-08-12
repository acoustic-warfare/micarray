function movie_test = movie_function(max_intensity,color_maps_movie,fps_rate,x_listen,y_listen)

f_bands_N = length(color_maps_movie(1,:));
movie1 = [];
movie2 = [];
temp_max1 = max_intensity*360000;
test12 = 0;
max_color_intensity = 0;

for frame_ind = 1:length(color_maps_movie(:,1))
    temp_map = zeros(length(y_listen),length(x_listen));
    for freq_ind = 1:f_bands_N
        temp_map = temp_map + color_maps_movie(frame_ind,freq_ind).color_data_matrix;
    end
    temp_max =  max(max(temp_map));
    if temp_max > max_color_intensity
        max_color_intensity = temp_max;
    end
    
end

for frame = 1:length(color_maps_movie(:,1))
    color_map_intensity = zeros(length(y_listen),length(x_listen));  
    color_maps_complete = color_maps_movie(frame,:);
    
    for freq_ind = 1:f_bands_N
        color_map_intensity =  color_map_intensity + color_maps_complete(freq_ind).color_data_matrix;
    end
    %temp_max = max(max(color_map_intensity));
    color_map_intensity = color_map_intensity/max_color_intensity;
    %color_map_intensity = color_map_intensity/temp_max;
    
    % Absolute value of the gradient of the spectrum map to obtain the sources locations
    
    % Calculate the gradient of the intensity (gradient_x, gradient_y), and the absolute value of the
    % gradient of the intensity (color_map_intensity_grad)
    color_map_intensity_grad = zeros(length(y_listen),length(x_listen));  
    gradient_x = zeros(length(y_listen),length(x_listen));
    gradient_y = zeros(length(y_listen),length(x_listen));
    
    for x_ind = 2:length(x_listen)-1
        for y_ind = 2:length(y_listen)-1
            % ( f(x+1) - f(x-1) )/2
            gradient_x(x_ind,y_ind) = (color_map_intensity(x_ind+1,y_ind) - ...
                color_map_intensity(x_ind-1,y_ind))/2;
    
            % ( f(y+1) - f(y-1) )/2
            gradient_y(x_ind,y_ind) = (color_map_intensity(x_ind,y_ind+1) - ...
                color_map_intensity(x_ind,y_ind-1))/2;
    
            gradient = [gradient_x(x_ind,y_ind); gradient_y(x_ind,y_ind);0];
            color_map_intensity_grad(x_ind,y_ind) = norm(gradient);
    
        end
    end
    
    % Calculate the divergence of the gradient (laplace operator) (color_map_intensity_grad2)
    color_map_intensity_grad2 = zeros(length(y_listen),length(x_listen));  
    for x_ind = 3:length(x_listen)-2
        for y_ind = 3:length(y_listen)-2
            % ( f(x+1) - f(x-1) )/2
            gradient_x_temp = (gradient_x(x_ind+1,y_ind) - ...
                gradient_x(x_ind-1,y_ind))/2;
    
            % ( f(y+1) - f(y-1) )/2
            gradient_y_temp = (gradient_y(x_ind,y_ind+1) - ...
                gradient_y(x_ind,y_ind-1))/2;
    
            color_map_intensity_grad2(x_ind,y_ind) = gradient_x_temp + gradient_y_temp;
        end
    end
    
    max_amp = max(max(color_map_intensity));
    
    grad2_max = max(max(color_map_intensity_grad2));
    grad_max = max(max(color_map_intensity_grad));
    intensity_max = max(max(color_map_intensity));
    
    
    % Precision map combines the intensity map, absolute derivative map and the
    % second derivative map to distinguish peaks
    %
    % By applying some threshold function to the precision_map, the location of
    % the sources seen by the audio_warfare algorithm can be obtained
    %
    
    precision_map_temp = ((1./(color_map_intensity_grad/grad_max) )).* -color_map_intensity_grad2/grad2_max.*color_map_intensity/intensity_max;
    precision_map = zeros(length(y_listen),length(x_listen));
    precision_map(2:end-1,2:end-1) = precision_map_temp(2:end-1,2:end-1);
    precision_map = precision_map./(max(max(precision_map)));

    treshold_precision_map = 0.85;

    for row = 1:length(precision_map(:,1))
        for column = 1:length(precision_map(1,:))
            if precision_map(row,column) < treshold_precision_map
                precision_map(row,column) = 0;
            end
        end
    end

    clim = [0 1];

    fig1 = figure(1);
    clf(fig1)
    imagesc(x_listen,y_listen,color_map_intensity,clim);
    set(gca,'YDir','normal')
    set(gca,'TickLabelInterpreter','latex','FontSize',12)
    title('Beamforming results',...
            'Interpreter','latex','FontSize',15);
    movie1 = [movie1 getframe;];

%     fig2 = figure(2);
%     clf(fig2)
%     imagesc(x_listen,y_listen,precision_map,clim);
%     set(gca,'YDir','normal')
%     set(gca,'TickLabelInterpreter','latex','FontSize',12)
%     title('Beamforming results enhanced',...
%             'Interpreter','latex','FontSize',15);
%     movie2 = [movie2 getframe];
end


%fig = figure(4);
clf(fig1)
movie(movie1,15,fps_rate);
%hold on
%movie(movie2,15,fps_rate-5);

movie_test = 0;

end

