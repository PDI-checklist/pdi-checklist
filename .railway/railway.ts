import { defineRailway, preserve, project, service, volume } from "railway/iac";

export default defineRailway(() => {
  const centralData = volume("pdi-central-data", {
    region: "asia-southeast1-eqsg1a",
    sizeMB: 1024,
  });

  const centralApi = service("pdi-central-api", {
    start: "uvicorn central_api:app --host 0.0.0.0 --port $PORT",
    healthcheck: "/health",
    healthcheckTimeout: 60,
    replicas: 1,
    env: {
      PDI_CENTRAL_API_TOKEN: preserve(),
      PDI_CENTRAL_DB_PATH: "/data/observations.sqlite3",
    },
    volumeMounts: {
      "/data": centralData,
    },
  });

  return project("pdi-phase2-mobile-upload", {
    resources: [centralApi, centralData],
  });
});
