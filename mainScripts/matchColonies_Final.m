function[] = matchColonies_Final(folderpath, time)
%The function is desing to match colonies over time after having extracting
%morphological characteristics from each time point. The tif data has to be
%organised by plate and include the different time points to be used.
%path = 'C:\Users\au648169\Documents\Postdoc_TorringLab\ImagingAnalysis\02-SecondRound\10%NBA(190exp)\PerPlate';

%Last version 25.05.2020

path = folderpath;
folders = dir('*nr*');
for a = 1:length(folders)
    subpath = strcat(path, '\', folders(a).name);
    cd(subpath);

    %clear all
    %close all

    files = dir('*data.mat');
    images = dir('*tif');
    %Better to read from the last image to the first
    %lst=sort(1:length(files), 'descend');
    lst=sort(1:length(files), 'ascend');

    %Extract the data to use in matching and growth rate
    endfiles = length(files);
    for i = 1:endfiles
        load(files(lst(i)).name);
        centroid{i} = statsData.centroid;
        diameters{i} = statsData.diameter;
    end 

    %%
    %Find the size of the contents of each array. Maximum value to pre-locate
    %memory
    cellsz = cellfun(@size,centroid,'uni',false);
    m = min(cell2mat(cellsz));
    n = length(centroid);

    %Match colonies base on their positions.
    for k = 1:n-1
        %Clean the index variable to save the new positions
        index={};
        found = [];
        %j=1;
        %Check the lenght of the current centroids list (k), if it is different
        %from the maximum m then the for loop will take the length of the
        %current list (k) otherwise it will take m.
        %if length(centroid{k}) ~= m
           % l = length(centroid{k});
       % else
            %
        %end 
        l = m;
        for i = 1:l
                %Check the y access within a range of +-10
                idy = find(centroid{k}(i,1)-10 <= centroid{k+1}(:,1) & centroid{k}(i,1)+10 >= centroid{k+1}(:,1));
                %If not found check withing a range of +-15
                if isempty(idy)
                    idy = find(centroid{k}(i,1)-15 <= centroid{k+1}(:,1) & centroid{k}(i,1)+15 >= centroid{k+1}(:,1));
                end
            %Check the x access within a range of +-10
                idx = find(centroid{k}(i,2)-10 <= centroid{k+1}(:,2) & centroid{k}(i,2)+10 >= centroid{k+1}(:,2));
                %If not found check withing a range of +-15
                if isempty(idx)
                    idx = find(centroid{k}(i,2)-15 <= centroid{k+1}(:,2) & centroid{k}(i,2)+15 >= centroid{k+1}(:,2));
                end

                %If find() empty, asume is not in the other plate and presetve
                %position. Important when the list k has more elements than k+1
                if isempty(idy) && isempty(idx)
                    %Assign empty value
                    index{i} = NaN;
                else
                    %If the searches are not empty, get the intersection, save
                    %the index only if the intersection length is 1 
                    id = intersect(idy, idx);
                    if length(id) == 1
                        index{i} = id;
                        %Keep track of the indexes that match, later it will be
                        %use to identify the new colonies in k+1
                        found = [found id];
                    else
                        index{i} = NaN;
                    end
                end
        end
    end
    
        %Turn the cell array with the re-ordered indexes to an array
        index = cell2mat(index);
        %Find the indexes not identified
        new = setdiff(1:length(centroid{k+1}(:,1)), found);
        %Build a new array with the re+-ordered and the new colonies from k+1
        indexN= [index new];
        %Save the indexes to use later
        %ID = indexN;

        %Re-order
        load(files(lst(k+1)).name);
        file = strsplit(files(lst(k+1)).name,'.');
        %Transform into table
        arrayfile = struct2array(statsData);

        n = length(indexN);
        %Create array. Since my data was all converted to strings, an empty
        %array of strings is needed.
        arraynew = strings(n, 36);
        for j = 1:n
            if isnan(indexN(j))
            else
                arraynew(j,:) = arrayfile(indexN(j),:);
            end
        end

        centroid{k+1} = [double(arraynew(:,4)), double(arraynew(:,5))];
        
        %%
        %Print new labels on plate
        colony = string(1:length(centroid{k+1}(:,1)));
        %Find the nan values
        tomap = ~isnan(centroid{k+1}(:,1));
        diameter = double(arraynew(:,7));
        
        %On figure
        I = imread(images(lst(k+1)).name);
        figure
        imshow(I);
        viscircles(centroid{k+1}(tomap,:),diameter(tomap)/2,'EdgeColor','b');
        text(centroid{k+1}(tomap,1), centroid{k+1}(tomap,2), colony(tomap));
        print(strcat(file{1},'-reIDs'),'-dpng');
        close;
        
%%        
        %Growth rate
        num1 = length(diameters{1});
        num2 = length(diameters{2});
        if num1 > num2
            num = num2;
        else
            num = num1;
        end
        diameter_t0 = zeros(length(diameter), 1);
        diameter_t0(1:num) = diameters{1}(1:num);
        growth_rate = abs(log(diameter) - log(diameter_t0))/(0.301*time);

%%

        %Save re-ordered data
        dataReordered = struct('label', arraynew(:,1), 'sample', arraynew(:,2), 'ID', arraynew(:,3),...
            'centroid', double(arraynew(:,4:5)), 'area', double(arraynew(:,6)), 'diameter_t1', diameter,...
            'diameter_t0', diameter_t0, 'growth_rate', growth_rate, 'perimeter', double(arraynew(:,8)),...
            'peaks', double(arraynew(:,9)), 'height_peaks', double(arraynew(:,10)), 'circularity', double(arraynew(:,11)),...
            'eccentricity', double(arraynew(:,12)),'RGB_mean', double(arraynew(:,13:15)),'RGB_std', double(arraynew(:,16:18)),...
            'RGBt_mean', double(arraynew(:,19:21)), 'RGBt_std', double(arraynew(:,22:24)),'Lab_mean', double(arraynew(:,25:27)),...
            'Lab_std', double(arraynew(:,28:30)), 'Labt_mean', double(arraynew(:,31:33)),'Labt_std', double(arraynew(:,34:36)));
        filename = strcat(path, '\',file{1},'-Reordered.mat');
        save(filename, 'dataReordered');
    
end    
end

