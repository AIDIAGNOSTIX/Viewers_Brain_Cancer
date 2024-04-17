window.config = {
  routerBasename: '/',
  whiteLabeling: {
    createLogoComponentFn: function(React) {
      return React.createElement('a', {
        target: '_self',
        rel: 'noopener noreferrer',
        className: 'header-brand',
        href: '/',
        style: {
          display: 'block',
          textIndent: '-9999px',
          background: 'url(../assets/aidiagnostix_logo.png)',
          backgroundSize: 'contain',
          backgroundRepeat: 'no-repeat',
          width: '200px',
        },
      });
    },
  },
  showStudyList: true,
  extensions: [],
  modes: [],
  // below flag is for performance reasons, but it might not work for all servers

  showWarningMessageForCrossOrigin: true,
  showCPUFallbackMessage: true,
  showLoadingIndicator: true,
  strictZSpacingForVolumeViewport: true,
  defaultDataSourceName: 'dicomweb',
  dataSources: [
    {
      namespace: '@ohif/extension-default.dataSourcesModule.dicomweb',
      sourceName: 'dicomweb',
      configuration: {
        dicomUploadEnabled: true,
        friendlyName: 'Orthanc Server',
        name: 'Orthanc',
        wadoUriRoot: 'http://127.0.0.1/pacs/dicom-web',
        qidoRoot: 'http://127.0.0.1/pacs/dicom-web',
        wadoRoot: 'http://127.0.0.1/pacs/dicom-web',
        qidoSupportsIncludeField: true,
        supportsReject: true,
        imageRendering: 'wadors',
        thumbnailRendering: 'wadors',
        enableStudyLazyLoad: true,
        supportsFuzzyMatching: true,
        supportsWildcard: true,
        dicomUploadEnabled: true,
        bulkDataURI: {
          enabled: true,
        },
      },
    },
    {
      namespace: '@ohif/extension-default.dataSourcesModule.dicomjson',
      sourceName: 'dicomjson',
      configuration: {
        friendlyName: 'dicom json',
        name: 'json',
      },
    },
    {
      namespace: '@ohif/extension-default.dataSourcesModule.dicomlocal',
      sourceName: 'dicomlocal',
      configuration: {
        friendlyName: 'dicom local',
      },
    },
  ],
};
