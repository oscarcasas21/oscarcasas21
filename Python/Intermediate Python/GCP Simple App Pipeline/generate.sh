#!/bin/sh
NOOFUSERS=100
ENDPOINTS=4
NOOFBTYPE=3
NOOFCTYPE=9

GENERATEREQS=1000

REQS=0
until [ $REQS -gt $GENERATEREQS ]; do
    ID=$(( ( RANDOM % $NOOFUSERS )  + 1 ))
    EP=$(( ( RANDOM % $ENDPOINTS )  + 1 ))
    BTYPE=$(( ( RANDOM % $NOOFBTYPE )  + 1 ))
    CTYPE=$(( ( RANDOM % $NOOFCTYPE )  + 1 ))
#    echo $ID $EP
    case $EP in
	1)
#    echo "sword"
	    case $BTYPE in
		1)
		    docker-compose exec mids curl "http://localhost:5000/purchase_sword/katana"
		    ;;
		2)
		    docker-compose exec mids curl "http://localhost:5000/purchase_sword/long-sword"
		    ;;
		3)
		    docker-compose exec mids curl "http://localhost:5000/purchase_sword/saber"
		    ;;
	    esac
	    ;;

	2)
#   	    echo "affiliation"
	    case $BTYPE in
		1)
		    docker-compose exec mids curl "http://localhost:5000/join_guild/nights"
		    ;;
		2)
		    docker-compose exec mids curl "http://localhost:5000/join_guild/vikings"
		    ;;
		3)
		    docker-compose exec mids curl "http://localhost:5000/join_guild/samurai"
		    ;;
	    esac
	    ;;
	3)
	    docker-compose exec mids curl -X POST "http://localhost:5000/login?id="$ID
	    ;;
	4)
	    docker-compose exec mids curl -X POST "http://localhost:5000/logout?id="$ID
	    ;;

    esac
    let REQS=REQS+1
done