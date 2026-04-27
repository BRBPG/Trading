module.exports = {
  apps: [{
    name: "trading-server",
    script: "server/index.js",
    cwd: "/root/Trading",
    node_args: "--experimental-specifier-resolution=node",
    env: {
      NODE_ENV: "production",
    },
    restart_delay: 5000,
    max_restarts: 10,
    watch: false,
  }]
};
