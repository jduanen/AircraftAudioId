#!/bin/bash
#
# Script to get information about a given aircraft identified with an ICAO hex id

HEX_ID=$1

curl -s https://opensky-network.org/api/metadata/aircraft/icao/${HEX_ID} | jq '.'
