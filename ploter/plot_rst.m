clear;clc;
%% Import data from text file.
% Script for importing data from the following text file:
%
%    /home/coyh4/code/RLCodes/DeepRL-0.3/log/dqn_pixel_atari-180821-171829/19d7d3a4-fca5-41a8-ba5f-f9faeaeef79e.monitor.csv
%
% To extend the code to different selected data or a different text file,
% generate a function instead of a script.

% Auto-generated by MATLAB on 2018/08/22 11:09:22

%% Initialize variables.
% Dueling DQN
%filename = '/home/coyh4/code/RLCodes/DeepRL-0.3/log/dqn_pixel_atari-180821-171829/19d7d3a4-fca5-41a8-ba5f-f9faeaeef79e.monitor.csv';
%filename = '/Users/andrea/work/4_Projects/DARPA/code/DeepRL-0.3/log/dqn_pixel_atari-180821-171829/19d7d3a4-fca5-41a8-ba5f-f9faeaeef79e.monitor.csv';

% Dueling DQN, with memory concatenated with the current observation
% filename = '/home/coyh4/code/RLCodes/DeepRL-0.3/log/dqn_pixel_atari-180916-175340/e667cc70-9fcd-4309-936e-bf86c1d4ec91.monitor.csv';
% filename = '/home/coyh4/code/RLCodes/DeepRL-0.3/log/dqn_pixel_atari-180826-185304/c8f65b29-9205-435f-9c6e-9df2c7edeac7.monitor.csv';
% filename = '/home/coyh4/code/RLCodes/DeepRL-0.3/log/dqn_pixel_atari-180830-175603/f16ec89a-a5bf-4ab0-9455-a931abbb00b9.monitor.csv';
% filename = '/home/coyh4/code/RLCodes/DeepRL-0.3/log/dqn_pixel_atari-180902-171958/12a834bb-1ada-4925-b2a3-ce01686a67e6.monitor.csv';

% Dueling DQN, softmax neural modulation applied to first conv layers
%filename = '/Users/andrea/work/4_Projects/DARPA/code/DeepRL-0.3/log/dqn_pixel_atari-180905-151536/a8d7370f-856d-401c-a38f-209b3de6a7fa.monitor.csv';

% Dueling DQN, softmax neural modulation applied to first two conv layers
%filename = '/Users/andrea/work/4_Projects/DARPA/code/DeepRL-0.3/log/dqn_pixel_atari-180909-183030/2c422352-68ef-44b5-a64d-7b77893b7fcb.monitor.csv';

% Dueling DQN, softmax neural modulation applied to first three conv layers

%2-layer mod
%filename = '/Users/andrea/work/4_Projects/DARPA/code/DeepRL-0.3/log/dqn_pixel_atari-180921-205456/cdca8e13-fa28-4ede-94ae-63e030b7348b.monitor.csv';
%2-layer mod? unsure
%filename = '/Users/andrea/work/4_Projects/DARPA/code/DeepRL-0.3/log/dqn_pixel_atari-180924-122210/a0f78458-9e18-45e3-a95b-df1fffa73fa2.monitor.csv';

%tmod2l-direct, no diff, sigmoid *0.5 + 1
%filename = '/Users/andrea/work/4_Projects/DARPA/code/DeepRL-0.3/log/dqn_pixel_atari-181003-153436/f18ab4db-f4fa-4b31-9e5a-2d0904e730ef.monitor.csv'

%tmod3l-direct: 1*sigmod, remaming (my first experiment)
%filename = '/Users/andrea/work/4_Projects/DARPA/code/DeepRL-0.3/log/dqn_pixel_atari-181003-221706/7bc65697-0109-483f-affc-35fa3e8f6507.monitor.csv';

%tmod2l-direct with 0.01 memory update
%filename = '/Users/andrea/work/4_Projects/DARPA/code/DeepRL-0.3/log/dqn_pixel_atari-181005-160107/0e708ea1-d61b-45e0-829b-75cbfd51cf7c.monitor.csv';

%tmod2l-direct with 0.05 memory update
%filename = '/Users/andrea/work/4_Projects/DARPA/code/DeepRL-0.3/log/dqn_pixel_atari-181005-160556/18acf200-3dcb-4fa4-a1a1-a68f31a5a59a.monitor.csv';


%tmod2lsurprise
filename = '/Users/andrea/work/4_Projects/DARPA/code/DeepRL-0.3/log/dqn_pixel_atari-181008-192752/52353637-f94e-4c8a-ad4b-0b60cf67281d.monitor.csv';

startRow = 3;

%% Format for each line of text:
%   column1: text (%s)
% For more information, see the TEXTSCAN documentation.
formatSpec = '%11s%[^\n\r]';

%% Open the text file.
fileID = fopen(filename,'r');

%% Read columns of data according to the format.
% This call is based on the structure of the file used to generate this
% code. If an error occurs for a different file, try regenerating the code
% from the Import Tool.
dataArray = textscan(fileID, formatSpec, 'Delimiter', '', 'WhiteSpace', '', 'TextType', 'string', 'EmptyValue', NaN, 'HeaderLines' ,startRow-1, 'ReturnOnError', false, 'EndOfLine', '\r\n');

%% Remove white space around all cell columns.
dataArray{1} = strtrim(dataArray{1});

%% Close the text file.
fclose(fileID);

%% Post processing for unimportable data.
% No unimportable data rules were applied during the import, so no post
% processing code is included. To generate code which works for
% unimportable data, select unimportable cells in a file and regenerate the
% script.

%% Create output variable
data = table;
data.env_id = cellstr(dataArray{:, 1});

%% Clear temporary variables
clearvars filename startRow formatSpec fileID dataArray ans;
data = table2cell(data);
num_step_interval = 100;
for i = 1:length(data)
    tmpData = data{i};
    idxSep = strfind(tmpData,',');
    reward(i) = str2num(tmpData(1:idxSep(1)-1));
    len(i) = str2num(tmpData(idxSep(1)+1:idxSep(2)-1));
end
lenth_total = cumsum(len);
% window_size = 1;
window_size = 100;
for i = window_size+1:length(data)
    reward_wnd(i-window_size) = mean(reward(i-window_size:i));
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
plot(lenth_total(window_size+1:end),reward_wnd);hold on;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% numPlot = 9000;
% plot(lenth_total(window_size+1:window_size+numPlot),reward_wnd(1:numPlot));hold on;
