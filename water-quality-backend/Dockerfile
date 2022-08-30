FROM jupyter/all-spark-notebook
WORKDIR /app
COPY taller_final.ipynb .
COPY water_potability.csv .
RUN pip install jupyter_kernel_gateway
RUN pip install imblearn
RUN pip install tensorflow
EXPOSE 8888
CMD ["jupyter", "kernelgateway", "--KernelGatewayApp.api=kernel_gateway.notebook_http", "--KernelGatewayApp.seed_uri=taller_final.ipynb", "--KernelGatewayApp.ip=0.0.0.0", "--KernelGatewayApp.allow_origin=*", "--KernelGatewayApp.port=8888"]