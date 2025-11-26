# base image
FROM quay.io/jupyter/minimal-notebook:afe30f0c9ad8

# copy lock file into container 

COPY conda-linux-64.lock /tmp/conda-linux-64.lock

# install packages from lock file
RUN conda create --name wine_quality_predictor_env --file /tmp/conda-linux-64.lock && conda clean --all -y

#activate the virtual environment by default
SHELL ["conda", "run", "-n", "wine_quality_predictor_env", "/bin/bash", "-c"]

# set working directory 
WORKDIR /home/jovyan/work

#expose jupyter lab port 
EXPOSE 8888

#default command 
CMD ["start-notebook.sh"]

