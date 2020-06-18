function[data] =dataCollectionGrowth_Final(filename)
%The function will collect into a single .mat file all the outputs from any
%ID function. Pay attention to the columns names since the only one that
%has that order so far is IDcfu_datapixelFinal_transversalcolonyPixel

%Last version 25.05.2020

%Get all the files in a directory
fnames = dir('*mat');


%Go to each one and gets do identification 
for k = 1:length(fnames)
    s = load(fnames(k).name);
    %Get the name of the structure
    StrName=fieldnames(s);
    StrName=StrName{1};
    if k == 1
        labelAll = s.(StrName).label;
        sampleAll = s.(StrName).sample;
        idAll = s.(StrName).ID;
        centroidAll = s.(StrName).centroid;
        areaAll = s.(StrName).area;
        diameter1All = s.(StrName).diameter_t1;
        diametert0All = s.(StrName).diameter_t0;
        growthAll = s.(StrName).growth_rate;
        perimeterAll = s.(StrName).perimeter;
        peaksAll = s.(StrName).peaks;
        height_peaksAll = s.(StrName).height_peaks;
        circularityAll = s.(StrName).circularity;
        eccentricityAll = s.(StrName).eccentricity;
        RGBmeanAll = s.(StrName).RGB_mean;
        RGBstdAll = s.(StrName).RGB_std;
        tRGBmeanAll = s.(StrName).RGBt_mean;
        tRGBstdAll = s.(StrName).RGBt_std;
        Lab_meanAll = s.(StrName).Lab_mean;
        Lab_stdAll = s.(StrName).Lab_std;
        Labt_meanAll = s.(StrName).Labt_mean;
        Labt_stdAll = s.(StrName).Labt_std;
       
    else
        labelAll = [labelAll; s.(StrName).label];
        sampleAll = [sampleAll; s.(StrName).sample];
        idAll = [idAll; s.(StrName).ID];
        centroidAll = [centroidAll; s.(StrName).centroid];
        areaAll = [areaAll; s.(StrName).area];
        diameter1All = [diameter1All; s.(StrName).diameter_t1];
        diametert0All = [diametert0All; s.(StrName).diameter_t0];
        growthAll = [growthAll; s.(StrName).growth_rate];
        perimeterAll = [perimeterAll; s.(StrName).perimeter];
        peaksAll = [peaksAll; s.(StrName).peaks];
        height_peaksAll = [height_peaksAll; s.(StrName).height_peaks];
        circularityAll = [circularityAll; s.(StrName).circularity];
        eccentricityAll = [eccentricityAll; s.(StrName).eccentricity];
        RGBmeanAll = [RGBmeanAll; s.(StrName).RGB_mean];
        RGBstdAll = [RGBstdAll; s.(StrName).RGB_std];
        tRGBmeanAll = [tRGBmeanAll; s.(StrName).RGBt_mean];
        tRGBstdAll = [tRGBstdAll; s.(StrName).RGBt_std];
        Lab_meanAll = [Lab_meanAll; s.(StrName).Lab_mean];
        Lab_stdAll = [Lab_stdAll; s.(StrName).Lab_std];
        Labt_meanAll = [ Labt_meanAll; s.(StrName).Labt_mean];
        Labt_stdAll = [Labt_stdAll;s.(StrName).Labt_std];
    end

end

%growthAll = transpose(growthAll);

%Save struct data
data = struct('label', labelAll, 'sample', sampleAll, 'ID', idAll,...
    'centroid', centroidAll, 'area', areaAll, 'diameter_t0', diametert0All,...
    'diameter_t1', diameter1All,'growth_rate', growthAll, 'perimeter', perimeterAll,...
    'peaks', peaksAll, 'height_peaks', height_peaksAll,...
    'circularity', circularityAll, 'eccentricity', eccentricityAll,...
    'RGB_mean', RGBmeanAll, 'RGB_std', RGBstdAll, 'RGBt_mean', tRGBmeanAll,...
    'RGBt_std', tRGBstdAll, 'Lab_mean', Lab_meanAll, 'Lab_std', Lab_stdAll,...
    'Labt_mean', Labt_meanAll, 'Labt_std', Labt_stdAll);
    save(strcat(filename,'-data.mat'), 'data');


%Save xls data
     tdata = struct2table(data);
     writetable(tdata, strcat(filename,'-data.xls'))
end


