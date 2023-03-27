clc
clear all
close all
warning off
FS=16000;%Sampling frequency in hertz
ch=1;%Number of channels--2 options--1 (mono) or 2 (stereo)
datatype='uint8';
nbits=16;%8,16,or 24
Nseconds=10;
% to record audio data from an input device ...
...such as a microphone for processing in MATLAB
recorder=audiorecorder(FS,nbits,ch);
disp('Start speaking..')
%Record audio to audiorecorder object,...
...hold control until recording completes
recordblocking(recorder,Nseconds);
disp('End of Recording.');
%Store recorded audio signal in numeric array
y=getaudiodata(recorder,datatype);
%Write audio file
audiowrite('180105172_SUCHI.wav',y,FS);