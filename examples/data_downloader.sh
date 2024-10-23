#!/bin/bash
# export API_TOKEN=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
export SERVER_URL=https://dataverse.harvard.edu
export PERSISTENT_ID=10.7910/DVN/CNMW2S
# export FILE_ID=4813200

curl -L -O -J $SERVER_URL/api/access/dataset/:persistentId?persistentId=doi:$PERSISTENT_ID

# curl -H "X-Dataverse-key:$API_TOKEN" $SERVER_URL/api/access/datafile/$FILE_ID

# wget https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/28075/K7L9Y8
