FROM jupyter/scipy-notebook:b72e40b2e3b1 as common

COPY fast-hgp fast-hgp

# Can't create build directories without it.
USER root
RUN cd fast-hgp && python3 -m pip install . 
USER ${NB_USER}

RUN mkdir /opt/conda/share/jupyter/lab/settings && echo '{ "@jupyterlab/apputils-extension:themes": { "theme": "JupyterLab Dark" } }' > /opt/conda/share/jupyter/lab/settings/overrides.json
