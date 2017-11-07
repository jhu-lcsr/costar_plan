
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
    local result = simSetObjectName(handle,string)
    if result == -1 then
      simAddStatusbarMessage('Setting object name failed: ' .. string)
    end
    simSetInt32Parameter(sim_intparam_error_report_mode,errorReportMode) -- restore the original error report mode
end

setObjectRelativeToParentWithPoseArray=function(handle, parent_handle, inFloats)
    -- Set the position and orientation of the parameter "handle" relative to the parent_handle
    -- infloats should be a table [x, y, z, qx, qy, qz, qw], though just 3 position arguments are acceptable
    if #inFloats>=3 then
      -- pose should be a vector with an optional quaternion array of floats
      -- 3 cartesian (x, y, z) and 4 quaternion (x, y, z, w) elements, same as vrep
      local result = simSetObjectPosition(handle, parent_handle, inFloats)
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
        local dummyHandle=-1
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
    -- Create a point cloud with specific name and coordinates, plus the option to clear an existing cloud.
    if #inStrings>=1 and #inInts>=8 and #inFloats>=1 then
        local cloudHandle=-1
        -- Get the existing point cloud's handle or create a new one
        if pcall(function()
            cloudHandle=simGetObjectHandle(inStrings[1])
        end) == false then
            -- create a new cloud if none exists
            local maxVoxelSize = inFloats[1]
            local max_point_count_per_voxel = inInts[6]
            local options = inInts[7]
            local pointSize = inInts[8]
            cloudHandle=simCreatePointCloud(maxVoxelSize, max_point_count_per_voxel, options, pointSize)
            -- Update the name of the cloud
            setObjectName(cloudHandle, inStrings[1])
        end

        local clearPointCloud=inInts[5]
        if clearPointCloud == 1 then
            simRemovePointsFromPointCloud(cloudHandle, 0, nil, 0)
        end
        -- Set the pose
        local parent_handle=inInts[1]
        return {cloudHandle},{},{},'' -- return the handle of the created dummy
    else
        simAddStatusbarMessage('createPointCloud_function call failed because incorrect parameters were passed.')
    end
end


insertPointCloud_function=function(inInts,inFloats,inStrings,inBuffer)
    -- inFloats contains the point cloud points
    -- inBuffer contains the pixel colors
    -- Create a dummy object with specific name and coordinates
    if #inStrings>=1 and #inInts>=6 then
        local cloudHandle=-1
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
        -- commented because we will set position and orientation separately later
        -- setObjectRelativeToParentWithPoseArray(cloudHandle, parent_handle, inFloats)
        -- Get the number of float entries used for the pose
        local poseEntries=inInts[2]
        -- number of floating point numbers in the point cloud
        local cloudFloatCount=inInts[3]
        -- bit 1 is 1 so point clouds in cloud reference frame
        local options = inInts[6]

        local colors = nil
        if options == 3 then
            colors = simUnpackUInt8Table(inBuffer)
        end
        -- Insert the points and color elements into the point cloud
        simInsertPointsIntoPointCloud(cloudHandle, options, inFloats, colors)
        return {cloudHandle},{},{},'' -- return the handle of the created dummy
    else
        simAddStatusbarMessage('createPointCloud_function call failed because incorrect parameters were passed.')
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
