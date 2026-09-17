FROM qiqcbench-qsim:latest
COPY . /tests
# Keep the image's expected inputs independent from Harbor's runtime ferry.
# The base-image alias would let an upload overwrite both comparison sources.
RUN test -L /app/configs && unlink /app/configs && mkdir /app/configs
# Keep the verifier container idle until Harbor execs /tests/test.sh.
ENTRYPOINT []
HEALTHCHECK NONE
