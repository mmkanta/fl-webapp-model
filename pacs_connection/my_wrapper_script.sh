#!/bin/bash
  
# turn on bash's job control
set -m


# Start the primary process and put it in the background
# python create_initial_file.py &&
# sleep 2 &&
while true; do
    [ -e stopme ] && break && sleep 2
    python start_scp_server.py
done

# python start_scp_server.py &

# Start the helper process or another process
# uvicorn main:app --port 8000 --host 0.0.0.0 &
# streamlit run app_V2_4.py --server.port 8501 

# starts the traffic app and restarts it if crashed
# while true; do
#     [ -e stopme ] && break
#     streamlit run app_V2_4.py --server.enableCORS False --server.port 8501
# done

# the my_helper_process might need to know how to wait on the
# primary process to start before it does its work and returns

  
# now we bring the primary process back into the foreground
# and leave it there
fg %1