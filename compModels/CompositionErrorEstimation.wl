(* ::Package:: *)

BeginPackage["CompositionErrorEstimation`"]

InitializeModel::usage = "InitializeModel[path, lDelay,eWidth,eHeight] loads the models CalCombo.mo and calCombo.m from the provided folder path and uses linear delay lDelay";
EvaluatePathPrediction::usage = "EvaluatePathPrediction[pathWithAllComps_] returns a list of the predicted ratio at each equivalent point for each composition. It is up to caller to process errors or other information";
EvaluatePathError::usage = "EvaluatePathError[pathWithAllComps, extrusionWidth, extrusionHeight] returns a list of total error per composition given the desired compositions at each point along a path's length. Path format: {{cumulativeLength,A,B,C}, {furtherPoints},...{finalPoint}} where every point is a control and desire";
EvaluatePathErrorWithEmpty::usage = "EvaluatePathErrorWithEmpty[pathShiftedCompositions_,pathDesiredCompositions_] returns a list of total error per composition at the lengths designated by pathDesiredCompositions given a path pathShiftedCompositions with compositions shifted into empty regions";
BatchEvalPathWEmpty::usage = "BatchEvalPathWEmpty[shiftedList_,desiredList_] returns a list of total error per composition for every path in shiftedList. desiredList should be the same length";

Begin["`Private`"]

loadedModel = Null;
(*calibratedModel = Null;*)
volumetricDelay = Null;
extrusionWidth = Null;
extrusionHeight = Null;

InitializeModel[modelFolderPath_, lDelay_, eWidth_, eHeight_]:= Module[{},
	loadedModel=Import[FileNameJoin[modelFolderPath,"CalCombo.mo"]];
(*	calibratedModel=Import[FileNameJoin[modelFolderPath,"calCombo.m"]];*)
	volumetricDelay = axisToCubic[lDelay];
	extrusionWidth = eWidth;
	extrusionHeight = eHeight;
	Return[loadedModel];
];

EvaluatePathPrediction[pathWithAllComps_]:=Module[{mixesByVolume, maxVolume, desiredMixSteps, delayDesiredMixSteps, predictions, errors},
	If[loadedModel === Null, Return[$Failed]];
	mixesByVolume=convertMixesByLenToMixByVol[pathWithAllComps];
	maxVolume=First[Last[First[mixesByVolume]]];
	desiredMixSteps=Table[StepFunction[ctlMixByV,Right],{ctlMixByV, mixesByVolume}];
	delayDesiredMixSteps=Table[StepFunction[Table[pt+{volumetricDelay,0},{pt,ctlMixByV}],Right],{ctlMixByV,mixesByVolume}];
	predictions=Table[predictedMixFunction[desiredMixSteps[[i]][-1],delayDesiredMixSteps[[i]],maxVolume,loadedModel],{i, Length[desiredMixSteps]}];
	Return[Table[predictions[[i]][mixesByVolume[[i,j,1]]],{i,Length[predictions]},{j,Length[mixesByVolume[[i]]]}]];
];

EvaluatePathError[pathWithAllComps_]:=Module[{mixesByVolume, maxVolume, desiredMixSteps, delayDesiredMixSteps, predictions, errors},
	If[loadedModel === Null, Return[$Failed]];
	mixesByVolume=convertMixesByLenToMixByVol[pathWithAllComps];
	maxVolume=First[Last[First[mixesByVolume]]];
	desiredMixSteps=Table[StepFunction[ctlMixByV,Right],{ctlMixByV, mixesByVolume}];
	delayDesiredMixSteps=Table[StepFunction[Table[pt+{volumetricDelay,0},{pt,ctlMixByV}],Right],{ctlMixByV,mixesByVolume}];
	predictions=Table[predictedMixFunction[desiredMixSteps[[i]][-1],delayDesiredMixSteps[[i]],maxVolume,loadedModel],{i, Length[desiredMixSteps]}];
	Return[Total[Table[predictions[[i]][mixesByVolume[[i,j,1]]]-mixesByVolume[[i,j,2]],{i,Length[predictions]},{j,Length[mixesByVolume[[i]]]}],{2}]];
];

EvaluatePathErrorWithEmpty[pathShiftedCompositions_,pathDesiredCompositions_]:=Module[{controlMixesByVolume, desireMixesByVolume, maxVolume, desiredMixSteps, delayControlMixSteps, predictions, errors},
	If[loadedModel === Null, Return[$Failed]];
	controlMixesByVolume=convertMixesByLenToMixByVol[pathShiftedCompositions];
	desireMixesByVolume=convertMixesByLenToMixByVol[pathDesiredCompositions];
	maxVolume=First[Last[First[desireMixesByVolume]]];
	desiredMixSteps=Table[StepFunction[ctlMixByV,Right],{ctlMixByV, desireMixesByVolume}];
	delayControlMixSteps=Table[StepFunction[Table[pt+{volumetricDelay,0},{pt,ctlMixByV}],Right],{ctlMixByV,controlMixesByVolume}];
	predictions=Table[predictedMixFunction[desiredMixSteps[[i]][-1],delayControlMixSteps[[i]],maxVolume,loadedModel],{i, Length[desiredMixSteps]}];
	Return[Total[Table[predictions[[i]][desireMixesByVolume[[i,j,1]]]-desireMixesByVolume[[i,j,2]],{i,Length[predictions]},{j,Length[desireMixesByVolume[[i]]]}],{2}]];
];

BatchEvalPathWEmpty[shiftedList_,desiredList_]:=Module[{errorList},
	errorList=Table[EvaluatePathErrorWithEmpty[shiftedList[[i]],desiredList[[i]]],{i,Length[shiftedList]}];
	Return[errorList];
];

cubicToAxis[mm3_]:=mm3/((1.75/2)^2*\[Pi]);
axisToCubic[mmE_]:=mmE * (1.75/2)^2*\[Pi];
lenToCubic[mmLen_,mmWidth_,mmHeight_]:=mmLen*mmWidth*mmHeight;
convertMixesByLenToMixByVol[mixesByLen_]:=Table[{lenToCubic[pos[[1]],extrusionWidth,extrusionHeight],pos[[mix]]},{mix,2,Length[mixesByLen[[1]]]},{pos,mixesByLen}];

predictedMixFunction[startMix_,offsetControlFunction_,maxVolume_,model_]:=
	SystemModelSimulate[model,
						{"m"},
						maxVolume,
						<|"InitialValues"->{"m"->startMix},"Inputs"->{"control"->offsetControlFunction}|>
	][{"m"}][[1]];

StepFunction[data_,pos_:Left]:=Module[{sdata,nf},sdata=Sort@data;
nf=Nearest[sdata[[All,1]]->"Index"];
StepFunction[nf,sdata[[All,1]],sdata[[All,2]],Replace[pos,{Left->{0,1},_->{-1,0}}]]];
	
(*Return a value*)
StepFunction[nf_NearestFunction,x_,y_,clip_][pt_List]:=With[{near=nf[pt][[All,1]]},y[[Clip[near+Clip[Sign[Subtract[pt,x[[near]]]],clip],{1,Length[x]}]]]];
StepFunction[nf_NearestFunction,x_,y_,clip_][pt_?NumericQ]:=With[{near=First@nf[pt]},y[[Clip[near+Clip[Sign[Subtract[pt,x[[near]]]],clip],{1,Length[x]}]]]];

(*make the objects look pretty as a box rather than unprocessed data*)
MakeBoxes[s:StepFunction[_,x_,y_,clip_],StandardForm]^:=Module[{left,g},left=MatchQ[clip,{0,1}];
g=Graphics[{Directive[Opacity[1.`],RGBColor[0.4`,0.5`,0.7`],AbsoluteThickness[1]],Line[{{0,0},{1,0},{1,1},{2,1}}],PointSize[Medium],If[left,Point[{{1,0},{2,1}}],Point[{{0,0},{1,1}}]]},Frame->True,FrameTicks->None,PlotRangePadding->{Scaled[.15],Scaled[.2]},AspectRatio->1,ImageSize->{Automatic,30}];
BoxForm`ArrangeSummaryBox[StepFunction,s,g,{BoxForm`MakeSummaryItem[{"Data points: ",Length[x]},StandardForm],BoxForm`MakeSummaryItem[{"Domain: ",x[[{1,-1}]]},StandardForm],BoxForm`MakeSummaryItem[{"Range: ",MinMax[y]},StandardForm]},{},StandardForm,"Interpretable"->True]];

End[] (*Closes private context*)
EndPackage[]
