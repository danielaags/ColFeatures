%%
%The script requires three different functions: i)ID_ function, ii)
%matching colonies time, iii)data_collection_matfiles. Functions should be 
%added in the main MATLAB folder to be easily accesible. Depending
%on the identificyation function use, the two other functions need to be 
%tailored to the specific output. Since the number of columns might change
%depending on first function used.
%%
%Inputs
%Day of the experiment
day = '200604';
%Name output file
filename_matfile = 'test';

%singleFolder = 1 for analysis of only one folder and singleFolder = 0 for 
%more than one folder
singleFolder = 0;
%If the experiment includes tracking along time (use two data points) use 1
%if you want to extract growth rate, else use 0
growth = 1;

%%
%Give the path where your folders are located
folderpath = pwd;

%%
%Single or multiple data points

%Single folder
if singleFolder == 1
    fnames = dir('*tif');
    numfids = length(fnames);

    %Go to each file
    for k = 1:numfids
      %Read the tif file
      file = strsplit(fnames(k).name,'.');
      plate =  num2str(k);
      %Run the desired ID function.
      IDcfu_Final(day, plate, file{1});
    end
    
    dataCollection_Final(filename_matfile);
%More than one folder
else
    %To check in more than one folder
    folders = dir('*nr*');
    for i = 1:length(folders)
        subpath = strcat(folderpath, '\', folders(i).name);
        cd(subpath);
        %Batch_idCFU('191123', i)
        %Get all the the tif files in a folder
        fnames = dir('*tif');
        numfids = length(fnames);
        plate =  num2str(i);

        %Go to each one and gets do identification 
        for k = 1:numfids
          %Read the tif file
          file = strsplit(fnames(k).name,'.');
          %Run the desired ID function.
          IDcfu_Final(day, plate, file{1});
        end
        if growth == 0
            dataCollection_Final(strcat(folders(i).name));
        end
        
    end
    if growth == 1    
        cd(folderpath);
        matchColonies_Final(folderpath,5);
    end
end

if growth == 1  
    cd(folderpath);
    dataCollectionGrowth_Final(filename_matfile);
end

