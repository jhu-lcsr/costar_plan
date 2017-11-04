
--- Improved functions based on remoteApiCommandServerExample.ttt
displayText_function=function(inInts,inFloats,inStrings,inBuffer)
    -- Simply display a dialog box that prints the text stored in inStrings[1]:
    if #inStrings>=1 then
        simAddStatusbarMessage('Message from the remote API client: ' .. inStrings[1])
        return {},{},{'message was displayed'},'' -- return a string
    end
end

setObjectName=function(handle, string)
    local errorReportMode=simGetInt32Parameter(sim_intparam_error_report_mode)
    simSetInt32Parameter(sim_intparam_error_report_mode,0) -- temporarily suppress error output (because we are not allowed to have two times the same object name)
    result = simSetObjectName(handle,string)
    if result == -1 then
      simAddStatusbarMessage('Setting object name failed: ' .. string)
    end
    simSetInt32Parameter(sim_intparam_error_report_mode,errorReportMode) -- restore the original error report mode
end

setObjectRelativeToParentWithPoseArray=function(handle, parent_handle, inFloats)
    if #inFloats>=3 then
      -- pose should be a vector with an optional quaternion array of floats
      -- 3 cartesian (x, y, z) and 4 quaternion (x, y, z, w) elements, same as vrep
      result = simSetObjectPosition(handle, parent_handle, inFloats)
      if #inFloats>=7 then
          local orientation={unpack(inFloats, 4, 7)} -- get 4 quaternion entries from 4 to 7
          result = simSetObjectQuaternion(handle, parent_handle, orientation)
      end
      return result
    end
end

createDummy_function=function(inInts, inFloats, inStrings, inBuffer)
    -- Create a dummy object with specific name and coordinates
    if #inStrings>=1 and #inFloats>=3 then
        dummyHandle=-1
        -- Get the existing dummy object's handle or create a new one
        if pcall(function()
            dummyHandle=simGetObjectHandle(inStrings[1])
        end) == false then
            dummyHandle=simCreateDummy(0.05)
            setObjectName(dummyHandle, inStrings[1])
        end

        -- Set the dummy position
        local parent_handle=inInts[1]
        setObjectRelativeToParentWithPoseArray(dummyHandle, parent_handle, inFloats)
        return {dummyHandle},{},{},'' -- return the handle of the created dummy
    end
end


createPointCloud_function=function(inInts,inFloats,inStrings,inBuffer)
    -- Create a dummy object with specific name and coordinates
    if #inStrings>=1 and #inFloats>=3 then
        cloudHandle=-1
        -- Get the existing point cloud's handle or create a new one
        if pcall(function()
            cloudHandle=simGetObjectHandle(inStrings[1])
        end) == false then
            -- create a new cloud if none exists
            cloudHandle=simCreatePointCloud(0.01, 10, 0, 10)
            -- Update the name of the cloud
            setObjectName(cloudHandle, inStrings[1])
        end

        -- Set the pose
        local parent_handle=inInts[1]
        setObjectRelativeToParentWithPoseArray(cloudHandle, parent_handle, inFloats)
        -- Get the number of float entries used for the pose
        poseEntries=inInts[2]
        -- print('pose vec quat:' .. {unpack(inFloats, 4, 7)})
        -- local cloud = simUnpackFloatTable(inStrings[2])
        cloudFloatCount=inInts[3]
        simAddStatusbarMessage('cloudFloatCount: '..cloudFloatCount)
        pointBatchSize=30
        colorBatch=nil
        -- bit 1 is 1 so point clouds in cloud reference frame
        options = 1
        if #inStrings > 2 then
          -- bit 2 is 1 so each point is colored
          options = 3
          colors = simUnpackUInt8Table(inStrings[3])
        end
        -- Insert the point cloud points
        for i = 1, cloudFloatCount, pointBatchSize do
            startEntry=1+poseEntries+i
            local pointBatch={unpack(inFloats, startEntry, startEntry+pointBatchSize)}
            simAddStatusbarMessage('threePoints:')

            simAddStatusbarMessage(pointBatch[1])
            simAddStatusbarMessage(pointBatch[2])
            simAddStatusbarMessage(pointBatch[3])
            if #inStrings > 2 then
                colorBatch = {unpack(colors, startEntry, startEntry+pointBatchSize)}
            end

           simInsertPointsIntoPointCloud(cloudHandle, options, pointBatch, colors)
        end
        return {cloudHandle},{},{},'' -- return the handle of the created dummy
    end
end

executeCode_function=function(inInts,inFloats,inStrings,inBuffer)
    -- Execute the code stored in inStrings[1]:
    if #inStrings>=1 then
        return {},{},{loadstring(inStrings[1])()},'' -- return a string that contains the return value of the code execution
    end
end

if (sim_call_type==sim_childscriptcall_initialization) then
    simExtRemoteApiStart(19999)
end
