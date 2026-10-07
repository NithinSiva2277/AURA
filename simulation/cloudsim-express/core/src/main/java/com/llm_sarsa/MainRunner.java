package com.llm_sarsa;

import org.cloudbus.cloudsim.Cloudlet;
import org.cloudbus.cloudsim.Datacenter;
import org.cloudbus.cloudsim.DatacenterBroker;
import org.cloudbus.cloudsim.DatacenterCharacteristics;
import org.cloudbus.cloudsim.Host;
import org.cloudbus.cloudsim.Log;
import org.cloudbus.cloudsim.Pe;
import org.cloudbus.cloudsim.Storage;
import org.cloudbus.cloudsim.UtilizationModelFull;
import org.cloudbus.cloudsim.Vm;
import org.cloudbus.cloudsim.VmAllocationPolicySimple;
import org.cloudbus.cloudsim.VmSchedulerTimeShared;
import org.cloudbus.cloudsim.core.CloudSim;
import org.cloudbus.cloudsim.provisioners.BwProvisionerSimple;
import org.cloudbus.cloudsim.provisioners.PeProvisionerSimple;
import org.cloudbus.cloudsim.provisioners.RamProvisionerSimple;

import java.util.ArrayList;
import java.util.Calendar;
import java.util.LinkedList;
import java.util.List;

public class MainRunner {
    public static void main(String[] args) {
        Log.printLine("Starting LLM-SARSA Cloud Orchestrator Sandbox...");

        try {
            CloudSim.init(1, Calendar.getInstance(), false);
            Datacenter datacenter = createDatacenter("Google_Cloud_Datacenter_1");
            DatacenterBroker broker = new DatacenterBroker("LLM_SARSA_Broker");
            int brokerId = broker.getId();

            List<Vm> vmlist = new ArrayList<Vm>();
            int[] mips = {50000, 25000, 5000, 100};
            for (int i = 0; i < mips.length; i++) {
                Vm vm = new Vm(i, brokerId, mips[i], 1, 512, 1000, 10000, "Xen", new LLMSARSAScheduler());
                vmlist.add(vm);
            }
            broker.submitVmList(vmlist);

            // 5. Create a workload of 25 Tasks to trigger SARSA learning
            List<Cloudlet> cloudletList = new ArrayList<Cloudlet>();
            for (int id = 0; id < 25; id++) {
                long length = 40000 + (long)(Math.random() * 60000); 
                Cloudlet cloudlet = new Cloudlet(
                    id, length, 1, 300, 300, 
                    new UtilizationModelFull(), new UtilizationModelFull(), new UtilizationModelFull()
                );
                cloudlet.setUserId(brokerId);
                cloudletList.add(cloudlet);
            }
            broker.submitCloudletList(cloudletList);

            CloudSim.startSimulation();
            CloudSim.stopSimulation();
            Log.printLine("Simulation finished!");

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static Datacenter createDatacenter(String name) throws Exception {
        List<Host> hostList = new ArrayList<Host>();
        List<Pe> peList = new ArrayList<Pe>();
        
        // HARDWARE FIX: 20 cores at 50,000 MIPS each
        for(int i=0; i<20; i++) peList.add(new Pe(i, new PeProvisionerSimple(50000))); 

        hostList.add(new Host(
            0, new RamProvisionerSimple(20480), new BwProvisionerSimple(10000), 1000000,
            peList, new VmSchedulerTimeShared(peList)
        ));

        DatacenterCharacteristics characteristics = new DatacenterCharacteristics(
            "x86", "Linux", "Xen", hostList, 10.0, 3.0, 0.05, 0.001, 0.0
        );

        return new Datacenter(name, characteristics, new VmAllocationPolicySimple(hostList), new LinkedList<Storage>(), 0);
    }
}