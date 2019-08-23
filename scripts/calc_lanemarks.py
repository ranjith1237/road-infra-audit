import json
from geopy.distance import geodesic
total=0
with open('coverage_score.json') as fp:
	fewframes_score=[]
	frame_coverage = json.load(fp)
	with open('data.json') as f:
		gps=json.load(f)
		cost=0
		data=gps["data"]
		for i in range(1,len(data)):
			old = (data[i-1]['longitude'],data[i-1]['latitude'])
			new = (data[i]['longitude'],data[i]['latitude'])
			dist=geodesic(old, new).meters
			total+=dist
			cost+=frame_coverage[str(i)][0]
			if total>=50:
				fewframes_score.append(cost/50)
				cost=0
				total=0
min_score = min(fewframes_score)
max_score = max(fewframes_score)
print(min_score,max_score)
normalized_scores=[]
print(fewframes_score)
for score in fewframes_score:
	norm_score = (score-min_score)/(max_score-min_score)
	normalized_scores.append(norm_score)
print(normalized_scores)
