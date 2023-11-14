# 先執行 Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
# 或 PowerShell.exe -ExecutionPolicy Bypass -File .\path\to\your\script.ps1

# Get the directory where the script file is located
$folderPath = Split-Path -Parent $MyInvocation.MyCommand.Definition

# Look for all .archimate files within the folder
$archimateFiles = Get-ChildItem -Path $folderPath -Filter *.archimate

# Process each file
foreach ($file in $archimateFiles) {
    # Load the XML content of the file
    [xml]$xml = Get-Content -Path $file.FullName -Raw -Encoding UTF8


# 创建一个命名空间管理器，并添加相应的命名空间
$nsManager = New-Object System.Xml.XmlNamespaceManager($xml.NameTable)
$nsManager.AddNamespace("xsi", "http://www.w3.org/2001/XMLSchema-instance")
$nsManager.AddNamespace("archimate", "http://www.archimatetool.com/archimate") # 替换为你的 XML 文件中定义的 archimate 命名空间 URI

# 使用命名空间管理器解析带有前缀的 XPath 查询
$elements = $xml.SelectNodes("//element[@xsi:type='archimate:BusinessActor']", $nsManager)

    foreach ($element in $elements) {
        # Check if the <element> has a <documentation> child with content starting with "file:///"
        $documentation = $element.SelectSingleNode("documentation[starts-with(text(), 'file:///')]")

        if ($documentation) {
            # Remove the existing <documentation> element
            $element.RemoveChild($documentation)
        }

        # Create a new <documentation> element
        $newDocumentation = $xml.CreateElement("documentation")
        $newDocumentation.InnerText = "http://192.168.0.20/ebc_ea/$($element.GetAttribute('name'))"

        # Append the new <documentation> element to the <element>
        $element.AppendChild($newDocumentation) > $null
    }

    # Save the changes back to the file
    $xml.Save($file.FullName)


    # 处理文件名以移除 "EBCEA_" 前缀和日期
    $fileNameWithoutExtension = [IO.Path]::GetFileNameWithoutExtension($file.Name)
    $reportFolderName = $fileNameWithoutExtension -replace "EBCEA_", "" -replace "[_\-\d-]*$", ""

    
    # 构造出 $folder 参数
    $reportFolder = Join-Path -Path $folderPath -ChildPath $reportFolderName


    # 构建命令行参数
    $arguments = "-consoleLog -console -nosplash -application com.archimatetool.commandline.app --html.createReport $reportFolder --loadModel $($file.FullName)"
    
    # 执行 Archi 命令行
    Start-Process -FilePath "C:\Program Files\Archi\Archi.exe" -ArgumentList $arguments -Wait -NoNewWindow
}


# 目标网络共享文件夹路径
$destination = "\\192.168.0.20\wwwroot\ebc_ea"


# 清空目标文件夹的内容
Get-ChildItem -Path $destination -Recurse | Remove-Item -Force -Recurse

# 从源文件夹复制内容到目标文件夹
Copy-Item -Path ("$folderPath\*") -Destination $destination -Recurse -Force


