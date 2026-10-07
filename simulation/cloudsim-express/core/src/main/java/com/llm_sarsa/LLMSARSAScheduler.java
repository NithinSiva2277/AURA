package com.llm_sarsa;

import org.cloudbus.cloudsim.CloudletSchedulerTimeShared;
import org.cloudbus.cloudsim.Cloudlet;
import org.cloudbus.cloudsim.DatacenterBroker;
import org.cloudbus.cloudsim.Vm;
import org.cloudbus.cloudsim.core.CloudSim;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.List;

public class LLMSARSAScheduler extends CloudletSchedulerTimeShared {

    private static final String PYTHON_API_URL = "http://localhost:8000/schedule_task";
    private final HttpClient httpClient;

    public LLMSARSAScheduler() {
        super();
        this.httpClient = HttpClient.newBuilder()
                .version(HttpClient.Version.HTTP_1_1)
                .connectTimeout(Duration.ofSeconds(10))
                .build();
    }

    @Override
    public double cloudletSubmit(Cloudlet cloudlet, double fileTransferTime) {
        long taskMi = cloudlet.getCloudletLength();
        int taskId = cloudlet.getCloudletId();
        int brokerId = cloudlet.getUserId();

        // 1. Retrieve the Broker and live VM List dynamically
        DatacenterBroker broker = (DatacenterBroker) CloudSim.getEntity(brokerId);
        List<Vm> vmList = broker.getVmList();

        // 2. Build the JSON array for the VMs (The Telemetry Sensor)
        StringBuilder vmsJson = new StringBuilder("[");
        for (int i = 0; i < vmList.size(); i++) {
            Vm vm = vmList.get(i);
            double cpuLoad = vm.getTotalUtilizationOfCpu(CloudSim.clock()) * 100.0;
            
            vmsJson.append(String.format(
                "{\"vm_id\": %d, \"cpu_capacity\": %s, \"ram_capacity\": %s, \"current_load_percent\": %s}",
                vm.getId(), String.valueOf(vm.getMips()), String.valueOf(vm.getRam()), String.valueOf(cpuLoad)
            ));
            
            if (i < vmList.size() - 1) {
                vmsJson.append(", ");
            }
        }
        vmsJson.append("]");

        // 3. Package the final JSON payload
        String jsonPayload = String.format(
            "{\"task_id\": %d, \"task_mi\": %d, \"task_type\": \"dynamic\", \"last_reward\": 0.0, \"vms\": %s}", 
            taskId, taskMi, vmsJson.toString()
        );

        // 4. Fire the HTTP POST request to Python
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(PYTHON_API_URL))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(jsonPayload))
                .build();

        try {
            HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
            System.out.println("[JAVA] Sent Task " + taskId + " to Python. AI Response: " + response.body());
            return super.cloudletSubmit(cloudlet, fileTransferTime);
        } catch (Exception e) {
            System.err.println("[JAVA] API Connection Failed: Is the Python FastAPI server running?");
            return super.cloudletSubmit(cloudlet, fileTransferTime);
        }
    }
}