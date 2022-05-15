export API_TOKEN=xxxxxxxxxxxxxxxxxxxxxxxxxx
export SERVER_URL=https://dataverse.harvard.edu
export PERSISTENT_ID=doi:10.7910/DVN/6ACUZJ

curl -L -O -J -H "X-Dataverse-key:$API_TOKEN" $SERVER_URL/api/access/dataset/:persistentId/?persistentId=$PERSISTENT_ID